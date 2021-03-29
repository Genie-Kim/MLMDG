from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import SGD
from tqdm import tqdm

from dataset.dg_dataset import *
from network.components.customized_evaluate import NaturalImageMeasure, MeterDicts
from network.components.schedulers import PolyLR
import segmodel
from utils.nn_utils import *
from utils.nn_utils import get_updated_network, get_logger, get_img_target
from utils.visualize import show_graphs

from network.nets.deeplabv3_plus import DeepLab

# 123456
seed = 123456
torch.manual_seed(seed)
np.random.seed(seed)


class MetaFrameWork(object):
    def __init__(self,network_init, name='normal_all', train_num=1, source='GSIM',
                 target='C', network='MemDeeplabv3plus', resume=True, dataset=DGMetaDataSets,
                 inner_lr=1e-3, outer_lr=5e-3, train_size=8, test_size=16, no_source_test=True,debug=False,lamb_cpt = 0.01,lamb_sep = 0.01,no_outer_memloss = False,no_inner_memloss = False,mem_after_update=True):
        super(MetaFrameWork, self).__init__()
        self.no_source_test = no_source_test
        self.train_num = train_num
        self.exp_name = name
        self.resume = resume

        self.inner_update_lr = inner_lr
        self.outer_update_lr = outer_lr

        if network in vars(segmodel):
            self.network = vars(segmodel)[network]
        else:
            raise NotImplementedError

        self.dataset = dataset
        self.train_size = train_size
        self.test_size = test_size
        self.source = source
        self.target = target

        self.epoch = 1
        self.best_target_acc = 0
        self.best_target_acc_source = 0
        self.best_target_epoch = 1

        self.best_source_acc = 0
        self.best_source_acc_target = 0
        self.best_source_epoch = 0

        self.total_epoch = 120
        self.save_interval = 1
        self.save_path = Path(self.exp_name)
        self.debug = debug

        self.lamb_cpt = lamb_cpt
        self.lamb_sep = lamb_sep
        self.no_outer_memloss = no_outer_memloss # if true, only segmentation loss on outer step.
        self.no_inner_memloss = no_inner_memloss # if true, only segmentation loss on inner step.
        self.mem_after_update = mem_after_update

        self.init(network_init)

        self.using_memory = True if network_init['memory_init'] != None else False



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

        dataloader = functools.partial(DataLoader, num_workers=workers, pin_memory=True, batch_size=self.test_size, shuffle=False)
        self.source_val_loader = dataloader(self.dataset(mode='val', domains=self.source, force_cache=True,crop_size=crop_size))

        target_dataset, folder = get_dataset(self.target)
        self.target_loader = dataloader(target_dataset(root=ROOT + folder, mode='val'))
        self.target_test_loader = dataloader(target_dataset(root=ROOT + 'cityscapes', mode='test'))

        self.opt_old = SGD(self.backbone.parameters(), lr=self.outer_update_lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler_old = PolyLR(self.opt_old, self.total_epoch, len(self.train_loader), 0, True, power=0.9)

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

        tr_net_output, tr_mem_output = self.backbone(meta_train_imgs)
        if self.using_memory:
            in_sep_loss = tr_mem_output[3]
            in_cpt_loss = tr_mem_output[2]
        else:
            in_cpt_loss = torch.zeros(1).cuda()
            in_sep_loss = torch.zeros(1).cuda()

        tr_logits = tr_net_output[0]
        tr_logits = make_same_size(tr_logits, meta_train_targets)
        in_seg_loss = self.ce(tr_logits, meta_train_targets[:, 0])

        if self.no_inner_memloss:
            ds_loss = in_seg_loss
        else:
            ds_loss = in_seg_loss + self.lamb_cpt * in_cpt_loss + self.lamb_sep * in_sep_loss

        with torch.no_grad():
            self.nim(tr_logits, meta_train_targets)

        self.opt_old.zero_grad()
        ds_loss.backward()
        self.opt_old.step()
        self.scheduler_old.step(epoch, it)

        # memory update.
        if self.using_memory:
            if self.mem_after_update: # memory updated by Et+1 encoder
                self.backbone.eval()
                with torch.no_grad():
                    self.backbone.module.update_memory(self.backbone(meta_train_imgs)[0][-1])
                self.backbone.train()
            else: # memory updated by mem_prime
                self.backbone.module.update_memory(tr_net_output[-1])

        losses = {
            'dg': 0,
            'ds': ds_loss.item()
        }
        acc = {
            'iou': self.nim.get_res()[0]
        }
        return losses, acc, self.scheduler_old.get_lr(epoch, it)[0]

    def meta_train(self, epoch, it, inputs):
        # imgs : batch x domains x C x H x W
        # targets : batch x domains x 1 x H x W

        imgs, targets = inputs
        B, D, C, H, W = imgs.size()
        split_idx = np.random.permutation(D)
        i = np.random.randint(1, D)
        train_idx = split_idx[:i]
        test_idx = split_idx[i:]
        # train_idx = split_idx[:D // 2]
        # test_idx = split_idx[D // 2:]

        # self.print(split_idx, B, D, C, H, W)'
        meta_train_imgs = imgs[:, train_idx].reshape(-1, C, H, W)
        meta_train_targets = targets[:, train_idx].reshape(-1, 1, H, W)
        meta_test_imgs = imgs[:, test_idx].reshape(-1, C, H, W)
        meta_test_targets = targets[:, test_idx].reshape(-1, 1, H, W)

        # Meta-Train

        tr_net_output, tr_mem_output = self.backbone(meta_train_imgs)
        if self.using_memory:
            in_sep_loss = tr_mem_output[3]
            in_cpt_loss = tr_mem_output[2]
        else:
            in_cpt_loss = torch.zeros(1).cuda()
            in_sep_loss = torch.zeros(1).cuda()

        tr_logits = tr_net_output[0]
        tr_logits = make_same_size(tr_logits, meta_train_targets)
        in_seg_loss = self.ce(tr_logits, meta_train_targets[:, 0])

        if self.no_inner_memloss:
            ds_loss = in_seg_loss
        else:
            ds_loss = in_seg_loss + self.lamb_cpt * in_cpt_loss + self.lamb_sep * in_sep_loss

        # Update new network
        self.opt_old.zero_grad()
        ds_loss.backward(retain_graph=True)
        self.updated_net = get_updated_network(self.backbone, self.updated_net, self.inner_update_lr).train().cuda()
        # synchronize memory
        self.updated_net.module.m_items = self.backbone.module.m_items

        # update inner updated network's memory using Et features
        with torch.no_grad():
            mem_prime = self.updated_net.module.update_memory(tr_net_output[-1])


        # Meta-Test
        te_net_output, te_mem_output = self.updated_net(meta_test_imgs)
        if self.using_memory:
            out_sep_loss = te_mem_output[3]
            out_cpt_loss = te_mem_output[2]
        else:
            out_cpt_loss = torch.zeros(1).cuda()
            out_sep_loss = torch.zeros(1).cuda()

        te_logits = te_net_output[0]
        te_logits = make_same_size(te_logits, meta_test_targets)
        out_seg_loss = self.ce(te_logits, meta_test_targets[:, 0])

        if self.no_outer_memloss:
            dg_loss = out_seg_loss
        else:
            dg_loss = out_seg_loss + self.lamb_cpt * out_cpt_loss + self.lamb_sep * out_sep_loss

        with torch.no_grad():
            self.nim(te_logits, meta_test_targets)

        # Update old network
        dg_loss.backward()
        self.opt_old.step()
        self.scheduler_old.step(epoch, it)

        # memory update.
        if self.using_memory:
            if self.mem_after_update: # memory updated by Et+1 encoder
                self.backbone.eval()
                with torch.no_grad():
                    tr_net_output, tr_mem_output = self.backbone(meta_train_imgs)
                    self.backbone.module.update_memory(tr_net_output[-1])
                self.backbone.train()
            else: # memory updated by mem_prime
                self.backbone.module.m_items = mem_prime

        losses = {
            'dg': dg_loss.item(),
            'ds': ds_loss.item(),

            'in_seg_loss' : in_seg_loss.item(),
            'in_cpt_loss' : in_cpt_loss.item(),
            'in_sep_loss' : in_sep_loss.item(),

            'out_seg_loss' : out_seg_loss.item(),
            'out_cpt_loss' : out_cpt_loss.item(),
            'out_sep_loss' : out_sep_loss.item(),
        }

        acc = {
            'iou': self.nim.get_res()[0],
        }

        return losses, acc, self.scheduler_old.get_lr(epoch, it)[0]

    def do_train(self):
        if self.resume:
            self.load()

        self.writer = SummaryWriter(str(self.save_path / 'tensorboard'), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))
        self.log('Start epoch : {}\n'.format(self.epoch))

        for epoch in range(self.epoch, self.total_epoch + 1):
            loss_meters, acc_meters = MeterDicts(), MeterDicts(averaged=['iou'])
            self.nim.clear_cache()
            self.backbone.train()
            self.epoch = epoch
            with self.train_timer:
                for it, (paths, imgs, target) in enumerate(self.train_loader):
                    meta = (it + 1) % self.train_num == 0
                    if meta:
                        losses, acc, lr = self.meta_train(epoch - 1, it, to_cuda([imgs, target]))
                    else:
                        losses, acc, lr = self.train(epoch - 1, it, to_cuda([imgs, target]))

                    loss_meters.update_meters(losses, skips=['dg'] if not meta else [])
                    acc_meters.update_meters(acc)

                    self.print(self.get_string(epoch, it, loss_meters, acc_meters, lr, meta), end='')
                    self.tfb_log(epoch, it, loss_meters, acc_meters)
                    if self.debug:
                        break
            self.print(self.train_timer.get_formatted_duration())
            self.log(self.get_string(epoch, it, loss_meters, acc_meters, lr, meta) + '\n')

            # self.save('ckpt')
            if epoch % self.save_interval == 0:
                with self.test_timer:
                    city_acc = self.val(self.target_loader)
                    self.save_best(city_acc, epoch)

            total_duration = self.train_timer.duration + self.test_timer.duration
            self.print('Time Left : ' + self.train_timer.get_formatted_duration(total_duration * (self.total_epoch - epoch)) + '\n')

        self.log('Best city acc : \n  city : {}, origin : {}, epoch : {}\n'.format(
            self.best_target_acc, self.best_target_acc_source, self.best_target_epoch))
        self.log('Best origin acc : \n  city : {}, origin : {}, epoch : {}\n'.format(
            self.best_source_acc_target, self.best_source_acc, self.best_source_epoch))

    def draw_tsne_domcls(self, path, ):
        self.load(path)
        self.backbone.eval()
        with torch.no_grad():
            for p, img, target in self.train_loader:
                img, target = to_cuda(get_img_target(img, target))
                if self.using_memory:
                    outputs = self.backbone(img,self.m_items,write=False)
                    fea = outputs[0][-1]
                    updated_fea = outputs[1][3]
                else:
                    outputs = self.backbone(img)
                    fea = outputs[0][-1]
                logits = outputs[0]
                self.nim(logits, target)
        self.log('\nNormal validation : {}\n'.format(self.nim.get_acc()))
        if hasattr(dataset.dataset, 'format_class_iou'):
            self.log(dataset.dataset.format_class_iou(self.nim.get_class_acc()[0]) + '\n')
        return self.nim.get_acc()[0]
        imgs, targets = inputs
        B, D, C, H, W = imgs.size()
        split_idx = np.random.permutation(D)
        i = np.random.randint(1, D)
        train_idx = split_idx[:i]
        test_idx = split_idx[i:]
        # train_idx = split_idx[:D // 2]
        # test_idx = split_idx[D // 2:]

        # self.print(split_idx, B, D, C, H, W)'
        meta_train_imgs = imgs[:, train_idx].reshape(-1, C, H, W)
        meta_train_targets = targets[:, train_idx].reshape(-1, 1, H, W)
        meta_test_imgs = imgs[:, test_idx].reshape(-1, C, H, W)
        meta_test_targets = targets[:, test_idx].reshape(-1, 1, H, W)

        # Meta-Train
        in_cpt_loss = torch.zeros(1).cuda()
        in_sep_loss = torch.zeros(1).cuda()

        if self.using_memory:
            tr_net_output, tr_mem_output = self.backbone(meta_train_imgs,keys = self.m_items)
            in_sep_loss = tr_mem_output.pop(-1)
            in_cpt_loss = tr_mem_output.pop(-1)
        else:
            tr_net_output, tr_mem_output = self.backbone(meta_train_imgs)
            for it, (paths, imgs, target) in enumerate(self.train_loader):
                meta = (it + 1) % self.train_num == 0
                if meta:
                    losses, acc, lr = self.meta_train(epoch - 1, it, to_cuda([imgs, target]))


    def save_best(self, city_acc, epoch):
        self.writer.add_scalar('acc/citys', city_acc, epoch)
        if not self.no_source_test:
            origin_acc = self.val(self.source_val_loader)
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
            for p, img, target in dataset:
                img, target = to_cuda(get_img_target(img, target))
                outputs = self.backbone(img)[0]
                logits = outputs[0]
                self.nim(logits, target)
        self.log('\nNormal validation : {}\n'.format(self.nim.get_acc()))
        if hasattr(dataset.dataset, 'format_class_iou'):
            self.log(dataset.dataset.format_class_iou(self.nim.get_class_acc()[0]) + '\n')
        return self.nim.get_acc()[0]

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

    # def predict_target(self, load_path='best_city', color=False, train=False, output_path='predictions'):
    #     self.load(load_path)
    #     import skimage.io as skio
    #     dataset = self.target_test_loader
    #
    #     output_path = Path(self.save_path / output_path)
    #     output_path.mkdir(exist_ok=True)
    #
    #     if train:
    #         self.backbone.module.remove_dropout()
    #         self.backbone.train()
    #     else:
    #         self.backbone.eval()
    #
    #     with torch.no_grad():
    #         self.nim.clear_cache()
    #         self.nim.set_max_len(len(dataset))
    #         for names, img, target in tqdm(dataset):
    #             img = to_cuda(img)
    #             logits = self.backbone(img)[0]
    #             logits = F.interpolate(logits, img.size()[2:], mode='bilinear', align_corners=True)
    #             preds = get_prediction(logits).cpu().numpy()
    #             if color:
    #                 trainId_preds = preds
    #             else:
    #                 trainId_preds = dataset.dataset.predict(preds)
    #
    #             for pred, name in zip(trainId_preds, names):
    #                 file_name = name.split('/')[-1]
    #                 if color:
    #                     pred = class_map_2_color_map(pred).transpose(1, 2, 0).astype(np.uint8)
    #                 skio.imsave(str(output_path / file_name), pred)

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


    def draw_tsne(self, inputimgname, load_path='best_city', output_path='tsne', perplexities=[30]):
        self.load(load_path)
        import skimage.io as skio
        dataset = self.target_test_loader

        output_path = Path(self.save_path / output_path)
        output_path.mkdir(exist_ok=True)

        self.backbone.eval()  # 이것으로 할 경우 moving average가 동작하지 않으며 statistic을 업데이트하지 않는다.
        sequence_of_colors = ["red", "green", "blue", "magenta", "cyan"]

        with torch.no_grad():
            self.nim.clear_cache()
            self.nim.set_max_len(len(dataset))
            for names, img, targets in tqdm(dataset):
                temp = False
                for name in names:
                    if inputimgname == name.split('/')[-1]:
                        temp = True
                if temp:
                    img = to_cuda(img)
                    feats = self.backbone(img)[
                        -1]  # resnet 50 에서 resnet block 네개지나고 끝에서 나온 feature를 conv layer에 넣어서 차원 축소 시킨 것.
                    targets = (F.interpolate(targets.type(torch.DoubleTensor), feats.size()[2:], align_corners=True,
                                             mode='bilinear')).type(
                        torch.uint8)  # down sampling for targets, 너무 크면 안돌아감 tsne
                    feats = feats.cpu().numpy()
                    targets = targets.cpu().numpy()
                    for feat, target, name in zip(feats, targets, names):  # 코드는 이상한데 test batch를 1ㄹ로 하면 되니까 오케이임.
                        if inputimgname == name.split('/')[-1]:
                            for perplexity in perplexities:

                                input_img = skio.imread(name)
                                file_name = os.path.splitext(name.split('/')[-1])[0]
                                input_img_name = file_name + '.jpg'
                                tsne_file_name = file_name + '_tsne_' + str(perplexity) + '.jpg'
                                skio.imsave(str(output_path / input_img_name), input_img)
                                feat = feat.reshape(512, -1).transpose()
                                target = target.reshape(1, -1).transpose().squeeze()
                                # X_embedded = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate).fit_transform(pt_tsne_X.cpu().numpy())
                                X_embedded = TSNE(n_components=2, perplexity=perplexity,
                                                  learning_rate=200).fit_transform(feat)  # use cpu memory
                                print('\ntsne done')
                                X_embedded = (X_embedded - X_embedded.min()) / (X_embedded.max() - X_embedded.min())
                                tsne_file_name = Path(output_path / tsne_file_name)
                                fig = plt.figure(figsize=(10, 10))
                                ax = fig.add_subplot(111)
                                label_set = set()
                                for i in range(X_embedded.shape[0]):
                                    label = dataset.dataset.idx_2_key[target[i]]
                                    if label not in label_set:
                                        label_set.add(label)
                                        ax.scatter(X_embedded[i, 0], X_embedded[i, 1],
                                                   c=sequence_of_colors[target[i]], label=label, s=6)
                                    else:
                                        ax.scatter(X_embedded[i, 0], X_embedded[i, 1],
                                                   c=sequence_of_colors[target[i]], s=6)
                                print('scatter plot done')

                                lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.1, 1))
                                ax.set_xlim(-0.05, 1.05)
                                ax.set_ylim(-0.05, 1.05)
                                fig.savefig(tsne_file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
                                fig.clf()
                            break
                    break

    def save_feature_per_domain(self, load_path='best_city', output_path='feature_domains', point_num=100,
                                size2d_feat=3):
        self.load(load_path)
        dataset = self.train_loader

        output_path = Path(Path(ROOT) / output_path)
        output_path.mkdir(exist_ok=True)

        self.backbone.eval()  # 이것으로 할 경우 moving average가 동작하지 않으며 statistic을 업데이트하지 않는다.
        name_sources = [x.root.name for x in dataset.dataset.domains]  # source name 순서대로 들어감, domain 폴더 이름이 들어감 순서지켜서.

        with torch.no_grad():
            self.nim.clear_cache()
            self.nim.set_max_len(len(dataset))
            pts_in_tsne = 0
            print('saving feature map per img, per domain, in disk.')
            while (True):
                for names, imgs, targets in tqdm(dataset):

                    imgs = to_cuda(imgs)

                    for i in range(len(name_sources)):
                        domain_name = name_sources[i]
                        img_in_bch = imgs[:, i, :]
                        domain_path = Path(output_path / domain_name)
                        domain_path.mkdir(exist_ok=True)

                        feats = self.backbone(img_in_bch)[
                            -1]  # resnet 50 에서 resnet block 네개지나고 끝에서 나온 feature를 conv layer에 넣어서 차원 축소 시킨 것.
                        feats = nn.AvgPool2d(math.ceil(feats.shape[-1] / size2d_feat)).forward(
                            feats)  # shrink feature map because increasing image number and limit of gpu memory for tsne

                        for feat, feat_num_per_domain in zip(feats, range(pts_in_tsne,
                                                                          pts_in_tsne + dataset.batch_size)):  # saving feature data per domain folder
                            feat_name = domain_name + str(feat_num_per_domain) + '.pth'
                            torch.save(feat.view(1, -1), Path(domain_path / feat_name))

                    pts_in_tsne = pts_in_tsne + dataset.batch_size  # data point per domain update
                    if pts_in_tsne > point_num:
                        break
                if pts_in_tsne > point_num:
                    break

    def draw_tsne_on_domains(self, output_path='feature_domains', perplexities=[30], learning_rate=10, point_num=100):
        assert perplexities is not list
        output_path = Path(Path(ROOT) / output_path)
        domainid2name = {}
        folder2alphabet = {
            'bdd_clear': 'C',
            'bdd_foggy': 'F',
            'bdd_rainy': 'R',
            'bdd_snowy': 'S',
            'bdd_overcast': 'O',
            'bdd_partlycloudy': 'P',
            'bdd_daytime': 'D',
            'bdd_dawndusk': 'W',
            'bdd_night': 'N',
        }
        # pt_tsne_X = to_cuda(torch.Tensor())
        sequence_of_colors = ["red", "green", "blue", "magenta", "cyan"]

        pt_tsne_X = torch.Tensor()  # use cpu memory
        pt_tsne_y = []
        source_names = []
        for domain_id, domain_name in enumerate(sorted(os.listdir(output_path))):
            domainid2name[domain_id] = domain_name
            domain_path = Path(output_path / domain_name)
            if os.path.isdir(domain_path) and folder2alphabet[domain_name] in self.source:
                source_names.append(domain_name)
                for i, feat_name in enumerate(tqdm(os.listdir(domain_path))):
                    feat = torch.load(Path(domain_path / feat_name))
                    feat = feat.cpu()  # use cpu memory
                    pt_tsne_X = torch.cat((pt_tsne_X, feat))
                    pt_tsne_y.append(domain_id)  # inplace method
                    if i > point_num:
                        break

        torch.cuda.empty_cache()
        for perplexity in perplexities:
            # X_embedded = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate).fit_transform(pt_tsne_X.cpu().numpy())
            X_embedded = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate).fit_transform(
                pt_tsne_X.numpy())  # use cpu memory
            print('\ntsne done')
            X_embedded = (X_embedded - X_embedded.min()) / (X_embedded.max() - X_embedded.min())
            tsne_file_name = 'feature_tsne_among_' + self.source + '_' + str(perplexity) + '_' + str(
                learning_rate) + '_' + str(point_num) + '_' + '.png'
            tsne_file_name = Path(output_path / tsne_file_name)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            label_set = set()
            for i in range(X_embedded.shape[0]):
                label = domainid2name[pt_tsne_y[i]]
                if label not in label_set:
                    label_set.add(label)
                    ax.scatter(X_embedded[i, 0], X_embedded[i, 1], c=sequence_of_colors[pt_tsne_y[i]], label=label, s=6)
                else:
                    ax.scatter(X_embedded[i, 0], X_embedded[i, 1], c=sequence_of_colors[pt_tsne_y[i]], s=6)
            print('scatter plot done')

            lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.1, 1))
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            fig.savefig(tsne_file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
            fig.clf()




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,0,1,2'
    framework = MetaFrameWork(name='exp', train_num=1, source='GSIM', target='C', debug=False, resume=True)
    framework.do_train()
    framework.val(framework.target_test_loader)
    from eval import test_one_run
    test_one_run(framework, 'previous_exps/dg_all', targets='C', batches=[16, 8, 1], normal_eval=False)
    framework.predict_target()
