import torch.nn as nn
import torchvision
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

class TVDeeplabRes101Encoder(nn.Module):
    """
    FCN-Resnet101 backbone from torchvision deeplabv3
    No ASPP is used as we found emperically it hurts performance
    """

    def __init__(self, aux_dim_keep=64, use_aspp=False):
        super().__init__()
        _model = torchvision.models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, progress=True,
                                                                     num_classes=21, aux_loss=None)


        _model_list = list(_model.children())
        self.aux_dim_keep = aux_dim_keep
        self.backbone = _model_list[0]
        self.localconv = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)  # reduce feature map dimension
        self.asppconv = nn.Conv2d(256, 256, kernel_size=1, bias=False)

        _aspp = _model_list[1][0]
        _conv256 = _model_list[1][1]
        self.aspp_out = nn.Sequential(*[_aspp, _conv256])
        self.use_aspp = use_aspp

    def forward(self, x_in, low_level):
        """
        Args:
            low_level: whether returning aggregated low-level features in FCN
        """
        fts = self.backbone(x_in)
        if self.use_aspp:
            fts256 = self.aspp_out(fts['out'])
            high_level_fts = fts256
        else:
            fts2048 = fts['out']
            high_level_fts = self.localconv(fts2048)

        if low_level:
            low_level_fts = fts['aux'][:, : self.aux_dim_keep]
            return high_level_fts, low_level_fts
        else:
            return high_level_fts
