# from network.utils import recursive_import_module
# recursive_import_module([__file__[:-12]], 'network.nets')
import network.nets.fcn
from .deeplabv3_plus import DeepLab
from .fcn import FCN8
from .unet import UNet, UNetResnet
from .segnet import SegNet
from .segnet import SegResNet
from .enet import ENet
from .gcn import GCN
from .deeplabv3_plus import DeepLab
from .duc_hdc import DeepLab_DUC_HDC
from .upernet import UperNet
from .pspnet import PSPNet
from .pspnet import PSPDenseNet