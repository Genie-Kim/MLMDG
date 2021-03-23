from torch import nn
from network.nets.SegNet import SegNet
from network.nets.deeplabv3_plus import DeepLab
from network.nets.Memory import Memory
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


class MemDeeplabv3plus(DeepLab):
    def __init__(self, in_ch=3, nclass=19, backbone='resnet50', output_stride=8, pretrained=True, freeze_bn=False,
                     freeze_backbone=False, memory_size=10, feature_dim=512, key_dim=512, temp_update=0.1,
                     temp_gather=0.1,skipconnect=True, **_):
        super(MemDeeplabv3plus, self).__init__(num_classes=nclass, in_channels=in_ch, backbone=backbone, pretrained=pretrained,
                output_stride=output_stride, freeze_bn=freeze_bn, freeze_backbone=freeze_backbone, **_)
        self.memory = Memory(memory_size,feature_dim, key_dim, temp_update, temp_gather)
        self.skipconnect = skipconnect

    def forward(self, x, keys, inner=True):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        fea = self.ASSP(x)
        if inner: # inner update : write, may cut skip connection

            # memory
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(
                fea, keys, inner)
            if self.skipconnect:
                output = self.decoder(updated_fea, low_level_features)
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)


        else: # outer update : do not write

            # memory
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(
                fea, keys, inner)

            if self.skipconnect:
                output = self.decoder(updated_fea, low_level_features)
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)

        return output, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss


    def not_track(self, module=None):
        if module is None:
            module = self
        if len(module._modules) != 0:
            for (k, v) in module._modules.items():
                self.not_track(v)
        else:
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False

    def remove_dropout(self):
        self.x[-1].p = 1e-10

    def recover_dropout(self):
        self.x[-1].p = 0.1