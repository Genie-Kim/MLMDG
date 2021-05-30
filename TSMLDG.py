import numpy as np
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import SGD
from tqdm import tqdm
from dataset import natural_datasets
from dataset.dg_dataset import *
from network.components.customized_evaluate import NaturalImageMeasure, MeterDicts
from network.components.schedulers import PolyLR
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import segmodel
from utils.nn_utils import *
from utils.nn_utils import get_updated_network, get_logger, get_img_target
from utils.visualize import show_graphs
import matplotlib.pyplot as plt
from dataset.transforms import Compose


from network.nets.deeplabv3_plus import DeepLab
from tsnecuda import TSNE

# 123456
seed = 123456
torch.manual_seed(seed)
np.random.seed(seed)


class MetaFrameWork(object):
    def __init__(self, param_dicts):

        super(MetaFrameWork, self).__init__()
        self.no_source_test = False
        self.train_num = param_dicts['train_num']
        self.name = param_dicts['name']
        self.exp_name = 'experiment/' + self.name
        self.resume = param_dicts['resume']

        self.inner_update_lr = param_dicts['inner_lr']
        self.outer_update_lr = param_dicts['outer_lr']

        self.backbone_name = param_dicts['network']
        if self.backbone_name in vars(segmodel):
            self.network = vars(segmodel)[self.backbone_name]
        else:
            raise NotImplementedError

        self.dataset = DGMetaDataSets
        self.train_size = param_dicts['train_size']
        self.test_size = param_dicts['test_size']
        self.source = param_dicts['source']
        self.target = param_dicts['target']

        self.epoch = 1
        self.best_target_acc = 0
        self.best_target_acc_source = 0
        self.best_target_epoch = 1

        self.best_source_acc = 0
        self.best_source_acc_target = 0
        self.best_source_epoch = 0

        self.save_interval = 1
        self.save_path = Path(self.exp_name)
        self.debug = param_dicts['debug']

        self.lamb_cpt = param_dicts['lamb_cpt']
        self.lamb_sep = param_dicts['lamb_sep']
        self.no_outer_memloss = param_dicts['no_outer_memloss'] # if true, only segmentation loss on outer step.
        self.no_inner_memloss = param_dicts['no_inner_memloss'] # if true, only segmentation loss on inner step.
        self.sche = param_dicts['sche']

        self.init(param_dicts['network_init'])

        self.using_memory = True if param_dicts['network_init']['memory_init'] != None else False


    def init(self,network_init):
        if self.debug:
            batch_size = 2
            workers = 0
            self.total_epoch = 3
            crop_size = 296
        else:
            batch_size = self.train_size
            workers = len(self.source) * 4
            crop_size = 600

        kwargs = network_init
        self.backbone = nn.DataParallel(self.network(**kwargs)).cuda()
        kwargs.update({'pretrained': False})
        self.updated_net = nn.DataParallel(self.network(**kwargs)).cuda()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.nim = NaturalImageMeasure(nclass=19)


        dataloader = functools.partial(DataLoader, num_workers=workers, pin_memory=True, batch_size=batch_size, shuffle=True)
        self.train_loader = dataloader(self.dataset(mode='train', domains=self.source, force_cache=True,crop_size=crop_size))

        dataloader = functools.partial(DataLoader, num_workers=workers, pin_memory=True, batch_size=self.train_size, shuffle=False)
        self.source_val_loader = dataloader(self.dataset(mode='val', domains=self.source, force_cache=True,crop_size=crop_size))

        target_dataset, folder = get_dataset(self.target)
        self.target_loader = dataloader(target_dataset(root=ROOT + folder, mode='test'))
        # if self.target != 'M':
        #     # deactivate target validation transform, domains have different img size. so target should be 1 on training.
        #     # if target datasets image size is not identical, should do validation transform.
        #     self.target_loader.dataset.transforms = None

        # self.target_test_loader = dataloader(target_dataset(root=ROOT + folder, mode='test'))

        self.opt_old = SGD(self.backbone.parameters(), lr=self.outer_update_lr, momentum=0.9, weight_decay=5e-4)


        if self.sche == 'lrplate':
            self.total_epoch = 200
            patience = 8 / self.save_interval
            self.scheduler_old = ReduceLROnPlateau(self.opt_old, factor=0.5, patience=patience,verbose = True)
        elif self.sche == 'poly':
            self.total_epoch = 200
            self.scheduler_old = PolyLR(self.opt_old, self.total_epoch, len(self.train_loader), 0, True, power=0.9)
        elif self.sche == 'cosine':
            T_0 = 80
            self.total_epoch = T_0 * 2
            minlr = self.outer_update_lr * 0.01
            self.scheduler_old = CosineAnnealingWarmRestarts(self.opt_old, T_0  = T_0, T_mult=1, eta_min = minlr)

        self.logger = get_logger('train', self.exp_name)
        self.log('exp_name : {}, train_num = {}, source domains = {}, target_domain = {}, lr : inner = {}, outer = {},'
                 'dataset : {}, net : {}\n'.
                 format(self.exp_name, self.train_num, self.source, self.target, self.inner_update_lr, self.outer_update_lr, self.dataset,
                        self.network))
        self.log(self.exp_name + '\n')
        self.train_timer, self.test_timer = Timer(), Timer()


    def train(self, epoch, it, inputs):
        # imgs : batch x domains x C x H x W
        # targets : batch x domains x 1 x H x W
        imgs, targets = inputs
        B, D, C, H, W = imgs.size()
        meta_train_imgs = imgs.view(-1, C, H, W)
        meta_train_targets = targets.view(-1, 1, H, W)

        # Meta-Train
        tr_net_output, _ = self.backbone(meta_train_imgs)
        tr_logits = tr_net_output[0]
        tr_logits = make_same_size(tr_logits, meta_train_targets)
        ds_loss = self.ce(tr_logits, meta_train_targets[:, 0])
        with torch.no_grad():
            self.nim(tr_logits, meta_train_targets)

        # Update new network
        self.opt_old.zero_grad()
        ds_loss.backward()
        self.opt_old.step()
        if self.sche=='poly':
            self.scheduler_old.step(epoch, it)

        losses = {
            'dg': 0,
            'ds': ds_loss.item(),
            'lr' : self.opt_old.param_groups[-1]['lr'],
        }

        acc = {
            'iou': self.nim.get_res()[0],
        }

        return losses, acc, self.opt_old.param_groups[-1]['lr']


    def train_mem(self, epoch, it, inputs):
        # imgs : batch x domains x C x H x W
        # targets : batch x domains x 1 x H x W
        imgs, targets = inputs
        B, D, C, H, W = imgs.size()
        meta_train_imgs = imgs.view(-1, C, H, W)
        meta_train_targets = targets.view(-1, 1, H, W)

        # backup the t_step memory items.
        backup_mem_t = self.backbone.module.m_items.clone().detach()

        # Meta-Train
        tr_net_output, tr_mem_output = self.backbone(meta_train_imgs,meta_train_targets,reading_detach = True)
        tr_logits = tr_net_output[0]
        tr_features = tr_net_output[-1]

        # update memory and get loss
        _, in_write_losses = self.backbone.module.update_memory(tr_features,meta_train_targets,writing_detach = False)

        tr_logits = make_same_size(tr_logits, meta_train_targets)
        in_seg_loss = self.ce(tr_logits, meta_train_targets[:, 0])
        in_read_loss = tr_mem_output[-2]

        if self.no_inner_memloss:
            ds_loss = in_seg_loss
        else:
            ds_loss = in_seg_loss + self.lamb_sep * (in_write_losses[0] + in_write_losses[1]) + self.lamb_cpt * in_read_loss

        with torch.no_grad():
            self.nim(tr_logits, meta_train_targets)

        # Update new network
        self.opt_old.zero_grad()
        ds_loss.backward()
        self.opt_old.step()
        if self.sche=='poly':
            self.scheduler_old.step(epoch, it)

        # memory update.
        self.backbone.eval()
        with torch.no_grad():
            # recover the t step memory.
            self.backbone.module.m_items = backup_mem_t
            # update memory to mem_t+1
            tr_net_output, _ = self.backbone(meta_train_imgs, reading_detach=True)
            tr_features = tr_net_output[-1]
            _, _ = self.backbone.module.update_memory(tr_features,meta_train_targets,writing_detach = True)
        self.backbone.train()

        losses = {
            'dg': 0,
            'ds': ds_loss.item(),

            'in_seg_loss' : in_seg_loss.item(),
            'in_write0_loss' : in_write_losses[0].item(),
            'in_write1_loss' : in_write_losses[1].item(),
            'in_read_loss' : in_read_loss.item(),

            'lr' : self.opt_old.param_groups[-1]['lr'],
        }

        acc = {
            'iou': self.nim.get_res()[0],
        }

        return losses, acc, self.opt_old.param_groups[-1]['lr']

    def meta_train(self, epoch, it, inputs,metridx, meteidx):
        # supervised mem + new update methods
        # imgs : batch x domains x C x H x W
        # targets : batch x domains x 1 x H x W
        imgs, targets = inputs
        B, D, C, H, W = imgs.size()
        # split_idx = np.random.permutation(D)
        # i = np.random.randint(1, D)
        # train_idx = split_idx[:i]
        # test_idx = split_idx[i:]
        train_idx = metridx
        test_idx = meteidx
        # train_idx = split_idx[:D // 2]
        # test_idx = split_idx[D // 2:]

        # self.print(split_idx, B, D, C, H, W)'
        meta_train_imgs = imgs[:, train_idx].reshape(-1, C, H, W)
        meta_train_targets = targets[:, train_idx].reshape(-1, 1, H, W)

        meta_test_imgs = imgs[:, test_idx].reshape(-1, C, H, W)
        meta_test_targets = targets[:, test_idx].reshape(-1, 1, H, W)

        # Meta-Train
        tr_net_output, _ = self.backbone(meta_train_imgs)
        tr_logits = tr_net_output[0]
        tr_logits = make_same_size(tr_logits, meta_train_targets)
        ds_loss = self.ce(tr_logits, meta_train_targets[:, 0])

        # Update new network
        self.opt_old.zero_grad()
        ds_loss.backward(retain_graph=True)
        self.updated_net = get_updated_network(self.backbone, self.updated_net, self.inner_update_lr).train().cuda()


        # Meta-Test
        te_net_output, _ = self.updated_net(meta_test_imgs)
        te_logits = te_net_output[0]
        te_logits = make_same_size(te_logits, meta_test_targets)
        dg_loss = self.ce(te_logits, meta_test_targets[:, 0])

        with torch.no_grad():
            self.nim(te_logits, meta_test_targets)

        # Update old network
        dg_loss.backward()
        self.opt_old.step()
        if self.sche=='poly':
            self.scheduler_old.step(epoch, it)

        losses = {
            'dg': dg_loss.item(),
            'ds': ds_loss.item(),
            'lr' : self.opt_old.param_groups[-1]['lr'],
        }

        acc = {
            'iou': self.nim.get_res()[0],
        }

        return losses, acc, self.opt_old.param_groups[-1]['lr']

    def meta_train_mem(self, epoch, it, inputs,metridx, meteidx):
        # supervised mem + new update methods
        # imgs : batch x domains x C x H x W
        # targets : batch x domains x 1 x H x W
        imgs, targets = inputs
        B, D, C, H, W = imgs.size()
        # split_idx = np.random.permutation(D)
        # i = np.random.randint(1, D)
        # train_idx = split_idx[:i]
        # test_idx = split_idx[i:]
        train_idx = metridx
        test_idx = meteidx
        # train_idx = split_idx[:D // 2]
        # test_idx = split_idx[D // 2:]

        # self.print(split_idx, B, D, C, H, W)'
        meta_train_imgs = imgs[:, train_idx].reshape(-1, C, H, W)
        meta_train_targets = targets[:, train_idx].reshape(-1, 1, H, W)

        meta_test_imgs = imgs[:, test_idx].reshape(-1, C, H, W)
        meta_test_targets = targets[:, test_idx].reshape(-1, 1, H, W)

        # backup the t_step memory items.
        backup_mem_t = self.backbone.module.m_items.clone().detach()

        # Meta-Train
        tr_net_output, tr_mem_output = self.backbone(meta_train_imgs,meta_train_targets,reading_detach = True)
        tr_logits = tr_net_output[0]
        tr_features = tr_net_output[-1]

        # update memory and get loss
        _, in_write_losses = self.backbone.module.update_memory(tr_features,meta_train_targets,writing_detach = False)

        tr_logits = make_same_size(tr_logits, meta_train_targets)
        in_seg_loss = self.ce(tr_logits, meta_train_targets[:, 0])
        in_read_loss = tr_mem_output[-2]

        if self.no_inner_memloss:
            ds_loss = in_seg_loss
        else:
            ds_loss = in_seg_loss + self.lamb_sep * (in_write_losses[0] + in_write_losses[1]) + self.lamb_cpt * in_read_loss

        # Update new network
        self.opt_old.zero_grad()
        ds_loss.backward(retain_graph=True)
        self.updated_net = get_updated_network(self.backbone, self.updated_net, self.inner_update_lr).train().cuda()
        # synchronize memory
        self.updated_net.module.m_items = self.backbone.module.m_items # mem_t'


        # Meta-Test
        te_net_output, te_mem_output = self.updated_net(meta_test_imgs,meta_test_targets,reading_detach = False)
        te_logits = te_net_output[0]
        te_logits = make_same_size(te_logits, meta_test_targets)
        out_seg_loss = self.ce(te_logits, meta_test_targets[:, 0])
        out_read_loss = te_mem_output[-2]

        if self.no_outer_memloss:
            dg_loss = out_seg_loss
        else:
            dg_loss = out_seg_loss + self.lamb_cpt * out_read_loss

        with torch.no_grad():
            self.nim(te_logits, meta_test_targets)

        # Update old network
        dg_loss.backward()
        self.opt_old.step()
        if self.sche=='poly':
            self.scheduler_old.step(epoch, it)

        # memory update.
        self.backbone.eval()
        with torch.no_grad():
            # recover the t step memory.
            self.backbone.module.m_items = backup_mem_t
            # update memory to mem_t+1
            tr_net_output, _ = self.backbone(meta_train_imgs, reading_detach=True)
            tr_features = tr_net_output[-1]
            _, _ = self.backbone.module.update_memory(tr_features,meta_train_targets,writing_detach = True)
        self.backbone.train()

        losses = {
            'dg': dg_loss.item(),
            'ds': ds_loss.item(),

            'in_seg_loss' : in_seg_loss.item(),
            'in_write0_loss' : in_write_losses[0].item(),
            'in_write1_loss' : in_write_losses[1].item(),
            'in_read_loss' : in_read_loss.item(),

            'out_seg_loss' : out_seg_loss.item(),
            'out_read_loss' : out_read_loss.item(),

            'lr' : self.opt_old.param_groups[-1]['lr'],
        }

        acc = {
            'iou': self.nim.get_res()[0],
        }

        return losses, acc, self.opt_old.param_groups[-1]['lr']


    def memory_initalize(self):
        self.backbone.eval()
        with torch.no_grad():
            basket = torch.zeros(size = self.backbone.module.m_items.size()).cuda()
            count = torch.zeros(size = (self.nim.nclass,1)).cuda()

            for epoch in range(2):
                for it, (path, imgs, targets) in enumerate(tqdm(self.train_loader,desc="memory initializing...epoch " + str(epoch))):
                    imgs, targets = to_cuda([imgs, targets])
                    B, D, C, H, W = imgs.size()
                    imgs = imgs.view(-1, C, H, W)
                    targets = targets.view(-1, 1, H, W)
                    query = self.backbone(imgs)[0][-1]
                    batch_size, dims, h, w = query.size()
                    targets = F.interpolate(targets.type(torch.float32), query.size()[2:], mode='bilinear',
                                             align_corners=True)
                    targets = targets.type(torch.int64)
                    query = query.view(batch_size, dims, -1)
                    targets[targets == -1] = self.nim.nclass  # when supervised memory, memory size = class number
                    targets = F.one_hot(targets, num_classes=self.nim.nclass + 1).squeeze()
                    targets = targets.view(batch_size, -1, self.nim.nclass + 1).type(torch.float32)

                    count += torch.t(targets.sum(1).unsqueeze(dim=1)[:,:,:self.nim.nclass].sum(0))
                    basket += torch.t(torch.matmul(query, targets)[:,:,:self.nim.nclass].sum(0))
                    if self.debug:
                        break
            count[count == 0] = 1 # for nan
            init_prototypes = torch.div(basket, count)
            self.backbone.module.m_items = F.normalize(init_prototypes, dim=1)

        self.backbone.train()

    def meta_transform(self, justidx = False):
        # this must called before dataloader enumerate.
        # meteidx is meta test dataset idx, and it will be hard augmentated
        D = len(self.source)
        split_idx = np.random.permutation(D)
        i = np.random.randint(1, D)
        metridx = split_idx[:i]
        meteidx = split_idx[i:]
        if ~justidx:
            for i in range(D):
                if i in metridx:
                    self.train_loader.dataset.domains[i].meta_transform = transforms_robust.RandomApply(
                        [transforms_robust.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)
                elif i in meteidx:
                    self.train_loader.dataset.domains[i].meta_transform = transforms_robust.RandomApply(
                        [transforms_robust.ColorJitter(0.8, 0.8, 0.8, 0.3)], p=0.5)
        return metridx, meteidx


    def do_train(self):
        if self.resume:
            self.load('best_city')
            # self.load()
        else:
            if self.using_memory:
                self.memory_initalize()

        self.writer = SummaryWriter(str(self.save_path / 'tensorboard'), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))
        self.log('Start epoch : {}\n'.format(self.epoch))

        for epoch in range(self.epoch, self.total_epoch + 1):
            loss_meters, acc_meters = MeterDicts(), MeterDicts(averaged=['iou'])
            self.nim.clear_cache()
            self.backbone.train()
            self.epoch = epoch
            with self.train_timer:
                metridx, meteidx = self.meta_transform(justidx=True)
                for it, (paths, imgs, target) in enumerate(self.train_loader):
                    meta = (it + 1) % self.train_num == 0
                    if meta:
                        if self.using_memory:
                            losses, acc, lr = self.meta_train_mem(epoch - 1, it, to_cuda([imgs, target]),metridx, meteidx)
                            # meta test idx domains get hard photometric augmentation.
                            metridx, meteidx = self.meta_transform()
                        else:
                            losses, acc, lr = self.meta_train(epoch - 1, it, to_cuda([imgs, target]),metridx, meteidx)
                    else:
                        if self.using_memory:
                            losses, acc, lr = self.train_mem(epoch - 1, it, to_cuda([imgs, target]))
                        else:
                            losses, acc, lr = self.train(epoch - 1, it, to_cuda([imgs, target]))
                    if lr < 5e-6:
                        assert False, 'No more training because small lr'

                    loss_meters.update_meters(losses, skips=['dg'] if not meta else [])
                    acc_meters.update_meters(acc)

                    self.print(self.get_string(epoch, it, loss_meters, acc_meters, lr, meta), end='')
                    self.tfb_log(epoch, it, loss_meters, acc_meters)
                    #torch.cuda.empty_cache()
                    if self.debug:
                        break
            torch.cuda.empty_cache()
            self.print(self.train_timer.get_formatted_duration())
            self.log(self.get_string(epoch, it, loss_meters, acc_meters, lr, meta) + '\n')

            # self.save('ckpt')
            if epoch % self.save_interval == 0:
                with self.test_timer:
                    city_acc, _ = self.val(self.target_loader)
                    self.save_best(city_acc, epoch)
            if self.sche == 'cosine':
                self.scheduler_old.step(epoch)

            total_duration = self.train_timer.duration + self.test_timer.duration
            self.print('Time Left : ' + self.train_timer.get_formatted_duration(total_duration * (self.total_epoch - epoch)) + '\n')

        self.log('Best city acc : \n  city : {}, origin : {}, epoch : {}\n'.format(
            self.best_target_acc, self.best_target_acc_source, self.best_target_epoch))
        self.log('Best origin acc : \n  city : {}, origin : {}, epoch : {}\n'.format(
            self.best_source_acc_target, self.best_source_acc, self.best_source_epoch))

    def save_best(self, city_acc, epoch):
        self.writer.add_scalar('acc/citys', city_acc, epoch)
        if not self.no_source_test:
            origin_acc,valloss = self.val(self.source_val_loader)
            if self.sche=='lrplate':
                self.scheduler_old.step(valloss)
            self.writer.add_scalar('acc/origin', origin_acc, epoch)
        else:
            origin_acc = 0

        self.writer.flush()
        if city_acc > self.best_target_acc:
            self.best_target_acc = city_acc
            self.best_target_acc_source = origin_acc
            self.best_target_epoch = epoch
            self.save('best_city')

        if origin_acc > self.best_source_acc:
            self.best_source_acc = origin_acc
            self.best_source_acc_target = city_acc
            self.best_source_epoch = epoch
            self.save('best_origin')

    def val(self, dataset):
        self.backbone.eval()
        with torch.no_grad():
            self.nim.clear_cache()
            self.nim.set_max_len(len(dataset))
            count = 0
            valloss = 0
            for p, img, target in dataset:
                img, target = to_cuda(get_img_target(img, target))
                outputs = self.backbone(img)[0]
                logits = outputs[0]
                valloss += self.ce(logits, target[:, 0])
                count += 1
                self.nim(logits, target)
        self.log('\nNormal validation : {}\n'.format(self.nim.get_acc()))
        if hasattr(dataset.dataset, 'format_class_iou'):
            self.log(dataset.dataset.format_class_iou(self.nim.get_class_acc()[0]) + '\n')
        return self.nim.get_acc()[0], valloss/count

    def target_specific_val(self, loader):
        self.nim.clear_cache()
        self.nim.set_max_len(len(loader))
        # eval for dropout
        self.backbone.module.remove_dropout()
        self.backbone.module.not_track()
        for idx, (p, img, target) in enumerate(loader):
            if len(img.size()) == 5:
                B, D, C, H, W = img.size()
            else:
                B, C, H, W = img.size()
                D = 1
            img, target = to_cuda([img.reshape(B, D, C, H, W), target.reshape(B, D, 1, H, W)])
            for d in range(img.size(1)):
                img_d, target_d, = img[:, d], target[:, d]
                self.backbone.train()
                with torch.no_grad():
                    new_logits = self.backbone(img_d)[0]
                    self.nim(new_logits, target_d)

        self.backbone.module.recover_dropout()
        self.log('\nTarget specific validation : {}\n'.format(self.nim.get_acc()))
        if hasattr(loader.dataset, 'format_class_iou'):
            self.log(loader.dataset.format_class_iou(self.nim.get_class_acc()[0]) + '\n')
        return self.nim.get_acc()[0]

    def get_string(self, epoch, it, loss_meters, acc_meters, lr, meta):
        string = '\repoch {:4}, iter : {:4}, '.format(epoch, it)
        for k, v in loss_meters.items():
            string += k + ' : {:.4f}, '.format(v.avg)
        for k, v in acc_meters.items():
            string += k + ' : {:.4f}, '.format(v.avg)

        string += 'lr : {:.6f}, meta : {}'.format(lr, meta)
        return string

    def log(self, strs):
        self.logger.info(strs)

    def print(self, strs, **kwargs):
        print(strs, **kwargs)

    def tfb_log(self, epoch, it, losses, acc):
        iteration = epoch * len(self.train_loader) + it
        for k, v in losses.items():
            self.writer.add_scalar('loss/' + k, v.val, iteration)
        for k, v in acc.items():
            self.writer.add_scalar('acc/' + k, v.val, iteration)

    def save(self, name='ckpt'):
        info = [self.best_source_acc, self.best_source_acc_target, self.best_source_epoch,
                self.best_target_acc, self.best_target_acc_source, self.best_target_epoch]
        dicts = {
            'backbone': self.backbone.state_dict(),
            'opt': self.opt_old.state_dict(),
            'epoch': self.epoch + 1,
            'best': self.best_target_acc,
            'info': info,
            'm_items': self.backbone.module.m_items
        }
        self.print('Saving epoch : {}'.format(self.epoch))
        torch.save(dicts, self.save_path / '{}.pth'.format(name))

    def load(self, path=None, strict=False):
        if path is None:
            path = self.save_path / 'ckpt.pth'
        else:
            if 'pth' in path:
                path = path
            else:
                path = self.save_path / '{}.pth'.format(path)

        try:
            dicts = torch.load(path, map_location='cpu')
            msg = self.backbone.load_state_dict(dicts['backbone'], strict=strict)
            self.print(msg)
            if 'opt' in dicts:
                self.opt_old.load_state_dict(dicts['opt'])
            if 'epoch' in dicts:
                self.epoch = dicts['epoch']
            else:
                self.epoch = 1
            if 'best' in dicts:
                self.best_target_acc = dicts['best']
            if 'info' in dicts:
                self.best_source_acc, self.best_source_acc_target, self.best_source_epoch, \
                self.best_target_acc, self.best_target_acc_source, self.best_target_epoch = dicts['info']
            if 'm_items' in dicts:
                if type(dicts['m_items']) == type(None):
                    print('There is not m_items in pth file. memory NOT updated')
                else:
                    self.backbone.module.m_items = dicts['m_items'].cuda()
                    print('memory items updated')
            self.log('Loaded from {}, next epoch : {}, best_target : {}, best_epoch : {}\n'
                     .format(str(path), self.epoch, self.best_target_acc, self.best_target_epoch))
            return True
        except Exception as e:
            self.print(e)
            self.log('No ckpt found in {}\n'.format(str(path)))
            self.epoch = 1
            return False

    def predict_target(self, load_path='best_city', output_path='predictions',savenum = 10,inputimgname=None):
        self.load(load_path)

        import skimage.io as skio
        dataset = self.target_loader
        self.target_loader.dataset.transforms = None

        output_path = Path(self.save_path / output_path)
        output_path.mkdir(exist_ok=True)
        overlay_rate = 0.7

        self.backbone.eval()


        with torch.no_grad():
            self.nim.clear_cache()
            self.nim.set_max_len(len(dataset))
            count = 1
            for names, img, target in tqdm(dataset):
                temp = False
                if inputimgname is not None:  # 특정 이미지에 대해서만 prediction하는 코드.
                    for name in names:
                        if inputimgname == name.split('/')[-1]:
                            temp = True
                elif count <= savenum:
                    temp = True
                else:
                    break

                if temp:
                    img = to_cuda(img)
                    logits = self.backbone(img)[0][0]
                    logits = F.interpolate(logits, img.size()[2:], mode='bilinear', align_corners=True)
                    trainId_preds = get_prediction(logits).cpu().numpy()
                    for pred, name, onelabel in zip(trainId_preds, names, target):
                        filename = name.split('/')[-1]
                        pred_file_name = os.path.splitext(filename)[0] + '_pred.jpg'
                        label_file_name = os.path.splitext(filename)[0] + '_label.jpg'
                        pred = class_map_2_color_map(pred).transpose(1, 2, 0).astype(np.uint8)
                        ori_img = skio.imread(name).astype(np.uint8)
                        onelabel = class_map_2_color_map(onelabel.cpu().numpy().squeeze()).transpose(1, 2, 0).astype(
                            np.uint8)
                        skio.imsave(str(output_path / pred_file_name),
                                    (1 - overlay_rate) * ori_img + overlay_rate * pred)
                        skio.imsave(str(output_path / label_file_name),
                                    (1 - overlay_rate) * ori_img + overlay_rate * onelabel)
                    count = count + 1
                else:
                    continue


    def draw_tsne_domcls(self, perplexities=[30], learning_rate=10, imagenum=1,pthname = 'ckpt.pth',output_path='tsne_domcls',all_class = False):

        assert perplexities is not list
        import skimage.io as skio
        overlay_rate = 0.7
        num_vec_imgcls = 100

        self.load(pthname)
        output_path = Path(self.save_path / output_path)
        output_path.mkdir(exist_ok=True)
        self.domains = (self.source + self.target)

        datasets = [(dom,get_target_loader(dom, 1, num_workers=0,mode='test',shuffle=False)) for dom in self.domains]

        if all_class:
            selected_cls = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
                      'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                      'bicycle']
        else:
            # selected_cls = ['road','building','vegetation','sky','person','car']
            selected_cls = ['road','building','vegetation','sky','person','car','bus']
            # selected_cls = ['person','vegetation','car']
            # selected_cls = ['road','building','sky']

        cls2id = self.target_loader.dataset.idx_of_objects

        self.backbone.eval()
        feat_vecs = torch.tensor([]).cuda()
        feat_vec_labels = torch.tensor([]).cuda()
        feat_vec_domlabels = torch.tensor([]).cuda()


        with torch.no_grad():
            for domi, (dom, dataset) in enumerate(tqdm(datasets)):
                count = 0
                for p, img, target in dataset:
                    img, target = to_cuda(get_img_target(img, target))
                    target_ori = target.clone()
                    outputs = self.backbone(img)[0]

                    features = outputs[-1]
                    features = F.normalize(features, dim=1)
                    target[target == -1] = self.nim.nclass
                    target = F.interpolate(target.to(torch.float32), features.size()[2:], mode='bilinear', align_corners=True)
                    target[(target - target.to(torch.int32).to(torch.float32)) != 0] = self.nim.nclass
                    mask = F.one_hot(target.to(torch.int64), num_classes=self.nim.nclass + 1).squeeze()

                    min_num_cls = True
                    for clsi, cls in enumerate(selected_cls): # 현재의 이미지에 class에 해당하는 vector의 개수가 최소 조건을 충족하는지 검사.
                        clsmask = mask[:,:,cls2id[cls]]
                        if (clsmask==1).sum() < num_vec_imgcls:
                            min_num_cls = False
                            break

                    if min_num_cls:
                        count += 1
                        logits = outputs[0]
                        logits = F.interpolate(logits, img.size()[2:], mode='bilinear', align_corners=True)
                        trainId_preds = get_prediction(logits).cpu().numpy()
                        # for pred, name, onelabel in zip(trainId_preds, p, target_ori): # feature 뽑은 이미지 prediction저장.
                        #     filename = name.split('/')[-1]
                        #     pred_file_name = os.path.splitext(filename)[0] + '_pred.jpg'
                        #     label_file_name = os.path.splitext(filename)[0] + '_label.jpg'
                        #     pred = class_map_2_color_map(pred).transpose(1, 2, 0).astype(np.uint8)
                        #     ori_img = skio.imread(name).astype(np.uint8)
                        #     onelabel = class_map_2_color_map(onelabel.cpu().numpy().squeeze()).transpose(1, 2, 0).astype(
                        #         np.uint8)
                        #     skio.imsave(str(output_path / pred_file_name),
                        #                 (1 - overlay_rate) * ori_img + overlay_rate * pred)
                        #     skio.imsave(str(output_path / label_file_name),
                        #                 (1 - overlay_rate) * ori_img + overlay_rate * onelabel)

                        for clsi,cls in enumerate(selected_cls):
                            clsmask = mask[:,:,cls2id[cls]]
                            cls_vec = torch.t(features[:, :, clsmask == 1].squeeze())

                            cls_vec = cls_vec[torch.randint(cls_vec.size()[0], (num_vec_imgcls,))] # sampled vector.

                            cls_label = torch.zeros(cls_vec.size()[0],1).cuda() + cls2id[cls]
                            dom_label = torch.zeros(cls_vec.size()[0],1).cuda() + domi

                            feat_vecs = torch.cat((feat_vecs, cls_vec), dim=0)
                            feat_vec_labels = torch.cat((feat_vec_labels, cls_label), dim=0)
                            feat_vec_domlabels = torch.cat((feat_vec_domlabels, dom_label), dim=0)

                    if count >= imagenum:
                        break

        if self.using_memory:
            m_items = self.backbone.module.m_items.clone().detach()
            mem_vecs = torch.tensor([]).cuda()
            mem_vec_labels = torch.tensor([]).cuda()

            # supervised.
            for clsi, cls in enumerate(selected_cls):
                mem_vecs = torch.cat((mem_vecs, m_items[cls2id[cls], :].unsqueeze(dim=0)), dim=0)
                mem_vec_labels = torch.cat((mem_vec_labels, torch.zeros(1,1).cuda() + cls2id[cls]), dim=0)


        del self.backbone
        torch.cuda.empty_cache()

        feat_vecs = feat_vecs.cpu().numpy()
        feat_vec_labels = feat_vec_labels.to(torch.int64).squeeze().cpu().numpy()
        feat_vec_domlabels = feat_vec_domlabels.to(torch.int64).squeeze().cpu().numpy()
        mem_vecs = mem_vecs.cpu().numpy()
        mem_vec_labels = mem_vec_labels.cpu().numpy()

        for perplexity in perplexities:

            # seen domain
            domains2draw = ['G', 'S']
            tsne_file_name = 'feature_tsne_among_' + ''.join(domains2draw) + '_' + str(perplexity) + '_' + str(
                learning_rate) + '.png'
            tsne_file_name = Path(output_path / tsne_file_name)
            self.draw_tsne(tsne_file_name, feat_vecs, feat_vec_labels, feat_vec_domlabels, mem_vecs, mem_vec_labels,
                           selected_cls, domains2draw=domains2draw, perplexity=perplexity, learning_rate=learning_rate)
            # unseen domain
            domains2draw = ['C']
            tsne_file_name = 'feature_tsne_among_' + ''.join(domains2draw) + '_' + str(perplexity) + '_' + str(
                learning_rate) + '.png'
            tsne_file_name = Path(output_path / tsne_file_name)
            self.draw_tsne(tsne_file_name, feat_vecs, feat_vec_labels, feat_vec_domlabels, mem_vecs, mem_vec_labels,
                           selected_cls, domains2draw=domains2draw, perplexity=perplexity, learning_rate=learning_rate)
            # all
            domains2draw = ['G', 'S', 'C']
            tsne_file_name = 'feature_tsne_among_' + ''.join(domains2draw) + '_' + str(perplexity) + '_' + str(
                learning_rate) + '.png'
            tsne_file_name = Path(output_path / tsne_file_name)
            self.draw_tsne(tsne_file_name, feat_vecs, feat_vec_labels, feat_vec_domlabels, mem_vecs, mem_vec_labels,
                           selected_cls, domains2draw=domains2draw, perplexity=perplexity, learning_rate=learning_rate)

            # domain wise tsne
            # tsne_file_name = 'feature_tsne_among_' + self.domains + '_' + str(perplexity) + '_' + str(
            #     learning_rate) + '_domain.png'
            # draw_tsne(tsne_file_name, feat_vec_domlabels, domains,feat_vecs)




    def draw_mean_tsne_domcls(self, perplexities=[30], learning_rate=10, imagenum=100,pthname = 'best_city.pth',output_path='tsne_domcls',all_class = False):
        # ToDo : 1. class feature 숫자가 균형맞지 않으면 tsne 그림이 이상해짐, class별 feature 숫자를 맞추고 여러 이미지에서 가져오는것 필요.
        #  2. 메모리 tsne 플롯한 결과 겹쳐서 나옴, 메모리 아이템을 저장할 때 노말라이즈를 하는데, backbone을 거쳐 나온 feature도 normalize를 하고 plot해야하는지 알아봐야함.

        assert perplexities is not list

        self.load(pthname)
        output_path = Path(self.save_path / output_path)
        output_path.mkdir(exist_ok=True)
        self.domains = (self.source + self.target)

        datasets = [(dom,get_target_loader(dom, 1, num_workers=0,mode='test',shuffle=True)) for dom in self.domains]

        if all_class:
            selected_cls = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
                      'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                      'bicycle']
        else:
            # selected_cls = ['road','building','vegetation','sky','person','car']
            selected_cls = ['road','building','vegetation','sky','person','car','bus']
            # selected_cls = ['person','vegetation','car']
            # selected_cls = ['road','building','sky']

        cls2id = self.target_loader.dataset.idx_of_objects
        selected_clsid = [cls2id[x] for x in selected_cls]

        self.backbone.eval()
        feat_vecs = torch.tensor([]).cuda()
        feat_vec_labels = torch.tensor([]).cuda()
        feat_vec_domlabels = torch.tensor([]).cuda()


        with torch.no_grad():
            for domi, (dom, dataset) in enumerate(datasets):
                count = 0
                for p, img, target in tqdm(dataset,dom + ' domain..'):
                    img, target = to_cuda(get_img_target(img, target))
                    outputs = self.backbone(img)[0]
                    features = outputs[-1]
                    features = F.normalize(features, dim=1)
                    target = F.interpolate(target.to(torch.float32), features.size()[2:], mode='bilinear',
                                           align_corners=True)
                    target[target == -1] = self.nim.nclass
                    target[(target - target.to(torch.int32).to(torch.float32)) != 0] = self.nim.nclass

                    tempmask = F.one_hot(target.to(torch.int64), num_classes=self.nim.nclass + 1).squeeze()
                    features = features.view(features.size(0), features.size(1), -1)
                    tempmask = tempmask.view(features.size(0), -1, self.nim.nclass + 1).type(torch.float32)
                    denominator = tempmask.sum(1).unsqueeze(dim=1)
                    nominator = torch.matmul(features, tempmask)

                    nominator = torch.t(nominator.sum(0))  # batchwise sum
                    denominator = denominator.sum(0)  # batchwise sum
                    denominator = denominator.squeeze()

                    for slot in range(self.nim.nclass):
                        if slot in selected_clsid:
                            if denominator[slot] != 0:
                                cls_vec = nominator[slot]/denominator[slot] # mean vector
                                cls_label = (torch.zeros(1,1) + slot).cuda()
                                dom_label = (torch.zeros(1,1) + domi).cuda()
                                feat_vecs = torch.cat((feat_vecs, cls_vec.unsqueeze(dim=0)), dim=0)
                                feat_vec_labels = torch.cat((feat_vec_labels, cls_label), dim=0)
                                feat_vec_domlabels = torch.cat((feat_vec_domlabels, dom_label), dim=0)
                    if count >= imagenum:
                        break
                    count += 1

        if self.using_memory:
            m_items = self.backbone.module.m_items.clone().detach()
            mem_vecs = torch.tensor([]).cuda()
            mem_vec_labels = torch.tensor([]).cuda()
            # supervised.
            for clsi, cls in enumerate(selected_cls):
                mem_vecs = torch.cat((mem_vecs, m_items[cls2id[cls], :].unsqueeze(dim=0)), dim=0)
                mem_vec_labels = torch.cat((mem_vec_labels, torch.zeros(1,1).cuda() + cls2id[cls]), dim=0)


        del self.backbone
        torch.cuda.empty_cache()

        feat_vecs = F.normalize(feat_vecs,dim=1).cpu().numpy()
        feat_vec_labels = feat_vec_labels.to(torch.int64).squeeze().cpu().numpy()
        feat_vec_domlabels = feat_vec_domlabels.to(torch.int64).squeeze().cpu().numpy()
        mem_vecs = mem_vecs.cpu().numpy()
        mem_vec_labels = mem_vec_labels.cpu().numpy()

        for perplexity in perplexities:

            # seen domain
            domains2draw = ['G', 'S']
            tsne_file_name = 'feature_tsne_among_' + ''.join(domains2draw) + '_' + str(perplexity) + '_' + str(
                learning_rate) + '.png'
            tsne_file_name = Path(output_path / tsne_file_name)
            self.draw_tsne(tsne_file_name, feat_vecs, feat_vec_labels, feat_vec_domlabels, mem_vecs, mem_vec_labels,
                           selected_cls, domains2draw=domains2draw, perplexity=perplexity, learning_rate=learning_rate)
            # unseen domain
            domains2draw = ['C']
            tsne_file_name = 'feature_tsne_among_' + ''.join(domains2draw) + '_' + str(perplexity) + '_' + str(
                learning_rate) + '.png'
            tsne_file_name = Path(output_path / tsne_file_name)
            self.draw_tsne(tsne_file_name, feat_vecs, feat_vec_labels, feat_vec_domlabels, mem_vecs, mem_vec_labels,
                           selected_cls, domains2draw=domains2draw, perplexity=perplexity, learning_rate=learning_rate)
            # all
            domains2draw = ['G', 'S', 'C']
            tsne_file_name = 'feature_tsne_among_' + ''.join(domains2draw) + '_' + str(perplexity) + '_' + str(
                learning_rate) + '.png'
            tsne_file_name = Path(output_path / tsne_file_name)
            self.draw_tsne(tsne_file_name, feat_vecs, feat_vec_labels, feat_vec_domlabels, mem_vecs, mem_vec_labels,
                           selected_cls, domains2draw=domains2draw, perplexity=perplexity, learning_rate=learning_rate)

            # domain wise tsne
            # tsne_file_name = 'feature_tsne_among_' + self.domains + '_' + str(perplexity) + '_' + str(
            #     learning_rate) + '_domain.png'
            # draw_tsne(tsne_file_name, feat_vec_domlabels, domains,feat_vecs)



    def draw_tsne(self,tsne_file_name, feat_vecs,feat_vec_labels,feat_vec_domlabels, mem_vecs,mem_vec_labels,cls_list,domains2draw = ['G','S'], perplexity = 30, learning_rate = 10): # per domain
        sequence_of_colors = ["tab:blue", "tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan","yellow" , "lawngreen","grey","darkslategray","lime","navy","blueviolet","blue","coral","peru", "lawngreen","grey","darkslategray","lime","navy","blueviolet"]
        domain_shape = ['o', '*', 'x', 'd']

        cls2id = self.target_loader.dataset.idx_of_objects

        domids2draw = [self.domains.index(x) for x in domains2draw]
        def split_dom_datas(domids2draw):
            temp = [feat_vec_domlabels == x for x in domids2draw]
            output = np.zeros(temp[0].shape,dtype=bool)
            for item in temp:
                output = output | item
            return feat_vecs[output], feat_vec_labels[output], feat_vec_domlabels[output]

        feat_vecs_temp, feat_vec_labels_temp, feat_vec_domlabels_temp = split_dom_datas(domids2draw)

        mem_address = feat_vecs_temp.shape[0]

        vecs2tsne = np.concatenate((feat_vecs_temp,mem_vecs))

        X_embedded = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate).fit_transform(
            vecs2tsne)
        print('\ntsne done')
        X_embedded = (X_embedded - X_embedded.min()) / (X_embedded.max() - X_embedded.min())

        feat_coords = X_embedded[:mem_address,:]
        mem_coords = X_embedded[mem_address:,:]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for dom_i,dom in enumerate(domids2draw):
            for cls_i,cls in enumerate(cls_list):
                temp_coords = feat_coords[(feat_vec_labels_temp == cls2id[cls]) & (feat_vec_domlabels_temp == dom),:]
                ax.scatter(temp_coords[:, 0], temp_coords[:, 1],
                           c=sequence_of_colors[cls_i], label=self.domains[dom]+'_'+cls, s=20, marker = domain_shape[dom_i])

        if self.using_memory:
            for cls_i,cls in enumerate(cls_list):
                ax.scatter(mem_coords[mem_vec_labels.squeeze() == cls2id[cls], 0], mem_coords[mem_vec_labels.squeeze() == cls2id[cls], 1],
                           c=sequence_of_colors[cls_i], label='mem_' + str(cls), s=100, marker=">",edgecolors = 'black')

        print('scatter plot done')
        lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.1, 1))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        fig.savefig(tsne_file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
        # plt.show()
        fig.clf()




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,0,1,2'
    framework = MetaFrameWork(name='exp', train_num=1, source='GSIM', target='C', debug=False, resume=True)
    framework.do_train()
    framework.val(framework.target_test_loader)
    from eval import test_one_run
    test_one_run(framework, 'previous_exps/dg_all', targets='C', batches=[16, 8, 1], normal_eval=False)
    framework.predict_target()