from typing import Callable, List
import torch
from torch import Tensor
from torch.nn import Module
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.models import resnet50, ResNet50_Weights


class CustomResNet(ResNet):
    def __init__(self, block,
                 layers: List[int], num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation = None,
                 norm_layer = None) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual,
                         groups, width_per_group, replace_stride_with_dilation, norm_layer)

        self.embed_dim = 2048

    def forward_features(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        bz, c, h, w = x.shape
        x = x.reshape(bz, c, h*w)
        x = x.permute(0, 2, 1)

        global_x = x.mean(dim=1)

        return global_x, x

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)


def get_resnet50(pretrained: bool = True):
    model = CustomResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict())

    return model


if __name__ == "__main__":
    resnet_50 = get_resnet50()

    x = torch.randn(1, 3, 512, 512)
    # print(resnet_50)
    global_x, local_x = resnet_50(x)
    print(global_x.shape)
    print(local_x.shape)
    # print(out.shape)
