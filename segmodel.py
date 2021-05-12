import torch
from torch import nn
from network.nets.deeplabv3_plus import DeepLab
from network.nets.Memory import Memory_unsup, Memory_sup
import torch.nn.functional as F

# class Net(SegNet):
#     def __init__(self, in_ch=3, nclass=19, backbone='resnet50', output_stride=8, pretrained=True, bn='torch'):
#         super(Net, self).__init__(in_ch, nclass, backbone, output_stride, pretrained, bn, multi_grid=False, aux=False)
#         self.x = nn.Sequential(
#             nn.Conv2d(2048, 2048 // 4, 3, padding=1, bias=False),
#             self.norm(2048 // 4),
#             nn.ReLU(),
#             nn.Dropout2d(0.1, False),
#         )
#         self.seg_classifier = nn.Conv2d(2048 // 4, nclass, 1)
#
#     def forward(self, x):
#         c1, c2, c3, c4 = self.backbone.base_forward(x)
#         feats = self.x(c4)
#         seg_logits = self.seg_classifier(feats)
#         return seg_logits, c1, c2, c3, c4, feats
#
#     def not_track(self, module=None):
#         if module is None:
#             module = self
#         if len(module._modules) != 0:
#             for (k, v) in module._modules.items():
#                 self.not_track(v)
#         else:
#             if isinstance(module, nn.BatchNorm2d):
#                 module.track_running_stats = False
#
#     def remove_dropout(self):
#         self.x[-1].p = 1e-10
#
#     def recover_dropout(self):
#         self.x[-1].p = 0.1


class Deeplabv3plus_Memunsup(DeepLab):
    def __init__(self, in_ch=3, nclass=19, backbone='resnet50', output_stride=8, pretrained=True, freeze_bn=False,
                     freeze_backbone=False, skip_connect = True, memory_init = None, **_):
        super(Deeplabv3plus_Memunsup, self).__init__(num_classes=nclass, in_channels=in_ch, backbone=backbone, pretrained=pretrained,
                output_stride=output_stride, freeze_bn=freeze_bn, freeze_backbone=freeze_backbone,skip_connect = skip_connect,**_)

        if type(memory_init) == dict:
            self.memory = Memory_unsup(**memory_init)
            self.m_items = F.normalize(torch.rand((memory_init['memory_size'], memory_init['feature_dim']), dtype=torch.float),
                                       dim=1).cuda()  # Initialize the memory items
        else:
            self.m_items = None


    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        fea = self.ASSP(x)
        features = F.normalize(fea, dim=1).clone().detach()
        mem_output = []
        # memory
        if type(self.m_items) != type(None):

            fea, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(
                fea, self.m_items)
            updated_features = fea.clone().detach()
            mem_output = [softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss,updated_features]

        fea = self.decoder(fea, low_level_features)

        output = F.interpolate(fea, size=(H, W), mode='bilinear', align_corners=True)

        return ([output,features], mem_output)


    def update_memory(self,query,mask=None):
        # update
        if type(self.m_items) != type(None):
            self.m_items = self.memory.update(query, self.m_items,mask)

        return self.m_items

    def get_memory(self):
        return self.m_items

    def not_track(self, module=None):
        if module is None:
            module = self
        if len(module._modules) != 0:
            for (k, v) in module._modules.items():
                self.not_track(v)
        else:
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False

class Deeplabv3plus_Memsup(DeepLab):
    def __init__(self, in_ch=3, nclass=19, backbone='resnet50', output_stride=8, pretrained=True, freeze_bn=False,
                     freeze_backbone=False, skip_connect = True, memory_init = None, add1by1 = False, **_):
        super(Deeplabv3plus_Memsup, self).__init__(num_classes=nclass, in_channels=in_ch, backbone=backbone, pretrained=pretrained,
                output_stride=output_stride, freeze_bn=freeze_bn, freeze_backbone=freeze_backbone,skip_connect = skip_connect,**_)

        if type(memory_init) == dict:
            self.memory = Memory_sup(**memory_init)
            self.m_items = F.normalize(torch.rand((memory_init['memory_size'], memory_init['feature_dim']), dtype=torch.float),
                                       dim=1).cuda()  # Initialize the memory items

            if add1by1:
                self.writefeat = nn.Sequential(  # refer object contextual represenation network fusion layer...
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
            else:
                self.writefeat = lambda x : x.clone()
        else:
            self.m_items = None


    def forward(self, x,mask=None,onlywriteloss=True):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        fea = self.ASSP(x)
        features = F.normalize(fea, dim=1).clone().detach()
        write_output = []
        # memory
        if type(self.m_items) != type(None):
            if type(mask) != type(None):
                writefeat = self.writefeat(fea)
                # update memory & calculate loss(writing)
                _, memloss = self.update_memory(writefeat,mask,onlywriteloss)
                write_output = [writefeat,memloss]
            # reading with updated memory
            fea, softmax_score_query, softmax_score_memory = self.memory(fea, self.m_items)
            updated_features = fea.clone().detach()
            read_output = [softmax_score_query, softmax_score_memory,updated_features]
            mem_output = read_output + write_output

        fea = self.decoder(fea, low_level_features)

        output = F.interpolate(fea, size=(H, W), mode='bilinear', align_corners=True)

        return ([output,features], mem_output)


    def update_memory(self,query,mask=None, onlyloss = False):
        # update
        # if onlyloss =True, then no update but calculate loss only.
        if type(self.m_items) != type(None):
            self.m_items, memloss = self.memory.update(query, self.m_items,mask,onlyloss)

        return self.m_items, memloss

    def get_memory(self):
        return self.m_items

    def not_track(self, module=None):
        if module is None:
            module = self
        if len(module._modules) != 0:
            for (k, v) in module._modules.items():
                self.not_track(v)
        else:
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False


class DeepLabv3plus(DeepLab):
    def __init__(self,in_ch=3, nclass=19, backbone='resnet50', output_stride=8, pretrained=True, freeze_bn=False,
                     freeze_backbone=False, **_):
        super(DeepLabv3plus, self).__init__(num_classes=nclass, in_channels=in_ch, backbone=backbone, pretrained=pretrained,
                output_stride=output_stride, freeze_bn=freeze_bn, freeze_backbone=freeze_backbone, **_)

    def not_track(self, module=None):
        if module is None:
            module = self
        if len(module._modules) != 0:
            for (k, v) in module._modules.items():
                self.not_track(v)
        else:
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False