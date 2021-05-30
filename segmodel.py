import torch
from torch import nn
from network.nets.deeplabv3_plus import DeepLab, ResNet
# from network.nets.resnet_fcn import fcn_resnet
from network.nets.Memory import Memory_unsup, Memory_sup
import torch.nn.functional as F
from dataset.transforms import HideAndSeek



class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)



class FCN_Memsup(nn.Module):
    def __init__(self, in_ch=3, nclass=19, backbone='resnet50', output_stride=8, pretrained=True,memory_init = None, hideandseek = True):
        super(FCN_Memsup, self).__init__()
        feat_channels =2048
        self.backbone = ResNet(in_channels=in_ch, output_stride=output_stride, pretrained=pretrained, backbone=backbone)

        if type(memory_init) == dict:
            self.reading_feat = nn.Sequential(
                nn.Conv2d(feat_channels, memory_init['feature_dim'], 1, padding=0, dilation=1, bias=False),
                nn.BatchNorm2d(memory_init['feature_dim']),
                nn.ReLU(inplace=True))

            self.fcnhead = FCNHead(memory_init['feature_dim'], nclass)

            if hideandseek:
                self.writeTF = HideAndSeek()
            else:
                self.writeTF = lambda x: x.clone()

            self.memory = Memory_sup(**memory_init)
            self.m_items = F.normalize(torch.rand((memory_init['memory_size'], memory_init['feature_dim']), dtype=torch.float),
                                       dim=1).cuda()  # Initialize the memory items
        else:
            self.fcnhead = FCNHead(feat_channels, nclass)
            self.m_items = None


    def forward(self, x,mask = None, reading_detach = True):
        H, W = x.size(2), x.size(3)
        fea = self.backbone(x)[0]
        features = fea.clone()
        read_output = []
        # memory
        if type(self.m_items) != type(None):
            fea = self.reading_feat(fea)
            features = fea.clone()
            # get reading loss
            reading_loss = 0
            if type(mask) != type(None):
                reading_loss = self.memory.get_reading_loss(self.m_items, fea, mask)
            # reading with updated memory
            fea, softmax_score_query, softmax_score_memory = self.memory(fea, self.m_items, reading_detach)
            updated_features = fea.clone()
            read_output = [softmax_score_query, softmax_score_memory, reading_loss, updated_features]

        fea = self.fcnhead(fea)
        output = F.interpolate(fea, size=(H, W), mode='bilinear', align_corners=True)

        return ([output,features], read_output)

    def update_memory(self, query, mask, writing_detach = True):
        # update
        if not writing_detach:
            query = self.writeTF(query) # only hide and seek for learning writing
        # if writing_detach = True, writing loss의 backpropagation은 되지만, 결과로 나온 updated memory의 gradient information은 없음.
        self.m_items, writing_losses = self.memory.update(query, self.m_items, mask, writing_detach)

        return self.m_items, writing_losses


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
                     freeze_backbone=False, skip_connect = True, memory_init = None,hideandseek = True, **_):
        super(Deeplabv3plus_Memsup, self).__init__(num_classes=nclass, in_channels=in_ch, backbone=backbone, pretrained=pretrained,
                output_stride=output_stride, freeze_bn=freeze_bn, freeze_backbone=freeze_backbone,skip_connect = skip_connect,**_)

        if type(memory_init) == dict:
            if hideandseek:
                self.writeTF = HideAndSeek()
            else:
                self.writeTF = lambda x: x.clone()

            self.memory = Memory_sup(**memory_init)
            self.m_items = F.normalize(torch.rand((memory_init['memory_size'], memory_init['feature_dim']), dtype=torch.float),
                                       dim=1).cuda()  # Initialize the memory items
        else:
            self.m_items = None


    def forward(self, x, mask = None, reading_detach = True):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        fea = self.ASSP(x)
        features = fea.clone()
        read_output = []
        # memory
        if type(self.m_items) != type(None):
            # get reading loss
            reading_loss = 0
            if type(mask) != type(None):
                reading_loss = self.memory.get_reading_loss(self.m_items, fea, mask)
            # reading with updated memory
            fea, softmax_score_query, softmax_score_memory = self.memory(fea, self.m_items,reading_detach)
            updated_features = fea.clone()
            read_output = [softmax_score_query, softmax_score_memory,reading_loss,updated_features]

        fea = self.decoder(fea, low_level_features)
        output = F.interpolate(fea, size=(H, W), mode='bilinear', align_corners=True)

        return ([output,features], read_output)

    def update_memory(self, query, mask, writing_detach = True):
        # update
        if not writing_detach:
            query = self.writeTF(query) # only hide and seek for learning writing
        # if writing_detach = True, writing loss의 backpropagation은 되지만, 결과로 나온 updated memory의 gradient information은 없음.
        self.m_items, writing_losses = self.memory.update(query, self.m_items, mask, writing_detach)

        return self.m_items, writing_losses


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
                     freeze_backbone=False, skip_connect= True, **_):
        super(DeepLabv3plus, self).__init__(num_classes=nclass, in_channels=in_ch, backbone=backbone, pretrained=pretrained,
                output_stride=output_stride, freeze_bn=freeze_bn, freeze_backbone=freeze_backbone,skip_connect=skip_connect, **_)


    def not_track(self, module=None):
        if module is None:
            module = self
        if len(module._modules) != 0:
            for (k, v) in module._modules.items():
                self.not_track(v)
        else:
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False

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

# class FCN(fcn_resnet):
#     def __init__(self, in_ch=3, nclass=19, backbone='resnet50', output_stride=8, pretrained=True):
#         if backbone == 'resnet50':
#             layers = [3, 4, 6, 3]
#         elif backbone == 'resnet101':
#             layers = [3, 4, 23, 3]
#         else:
#             assert False, 'no FCN backbone'
#
#         super(FCN, self).__init__(Bottleneck, layers, module_type=str(output_stride)+'s', n_classes=nclass, pretrained=pretrained, upsample_method='upsample_bilinear')
#         self.init_weight('fcn_'+backbone)
#
#
#
#     def forward(self, x):
#         x_size = x.size()[2:]
#         x_conv1 = self.conv1(x)
#         x = self.bn1(x_conv1)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x_layer1 = self.layer1(x)
#         x_layer2 = self.layer2(x_layer1)
#         x_layer3 = self.layer3(x_layer2)
#         x = self.layer4(x_layer3)
#
#         # x = self.avgpool(x)
#         # x = x.view(x.size(0), -1)
#         # x = self.fc(x)
#         score = self.classifier(x)
#
#         if self.module_type=='16s' or self.module_type=='8s':
#             score_pool4 = self.score_pool4(x_layer3)
#         if self.module_type=='8s':
#             score_pool3 = self.score_pool3(x_layer2)
#
#         if self.module_type=='16s' or self.module_type=='8s':
#             # print('score_pool4.size():', score_pool4.size())
#             # print('score.size():', score.size())
#             if self.upsample_method == 'upsample_bilinear':
#                 score = F.upsample_bilinear(score, score_pool4.size()[2:])
#             elif self.upsample_method == 'ConvTranspose2d':
#                 score = self.upsample_1(score)
#             score += score_pool4
#         if self.module_type=='8s':
#             # print('score_pool3.size():', score_pool3.size())
#             # print('score.size():', score.size())
#             if self.upsample_method == 'upsample_bilinear':
#                 score = F.upsample_bilinear(score, score_pool3.size()[2:])
#             elif self.upsample_method == 'ConvTranspose2d':
#                 score = self.upsample_2(score)
#             score += score_pool3
#
#         out = F.upsample_bilinear(score, x_size)
#
#         return out
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