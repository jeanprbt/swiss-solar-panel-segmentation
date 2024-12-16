import torch.nn as nn
import torchvision.models as models

class ResNet_50(nn.Module):
    """
    ResNet-50 model with pretrained weights.
    """
    def __init__(self, output_layer=None):
        super(ResNet_50, self).__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(*self.pretrained._modules.values())
        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x

