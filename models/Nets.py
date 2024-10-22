from torch import nn
from collections import OrderedDict
import torchvision.models as models
from models.resnet import resnet8

class Encoder_ResNet(nn.Module):
    def __init__(self, model_type):
        super(Encoder_ResNet, self).__init__()

        if 'resnet8' in model_type:
            model = resnet8()
        elif 'resnet18' in  model_type:
            model = models.resnet18(pretrained=True)
        elif 'resnet34' in model_type:
            model = models.resnet34(pretrained=True)
        elif 'resnet50' in model_type:
            model = models.resnet50(pretrained=True)
        elif 'resnet101' in model_type:
            model = models.resnet101(pretrained=True)
        elif 'resnet152' in model_type:
            model = models.resnet152(pretrained=True)
        else:
            model = models.resnet18(pretrained=True)

        # remove the last classification layer of ResNet
        new_layers = OrderedDict()
        for name, module in model.named_children():
            if name == 'fc':
                continue
            new_layers[name] = module
        self.feature1 = nn.Sequential(new_layers)

    def forward(self, x):
        f1 = self.feature1(x)
        f1 = f1.view(-1, f1.shape[1] * f1.shape[2] * f1.shape[3])
        return f1


class ResNet(nn.Module):
    def __init__(self, model_type, num_classes):
        super(ResNet, self).__init__()

        self.global_encoder = Encoder_ResNet(model_type)

        self.local_projector = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(512, 256)),
            ('relu_1', nn.ReLU()),
            ('bn_1', nn.BatchNorm1d(256)),
            ('linear_2', nn.Linear(256, 512)),
            ('bn_2', nn.BatchNorm1d(512)),
        ]))

        self.local_C = nn.Linear(512, num_classes)
        self.global_C = nn.Linear(512, num_classes)

    def forward(self, x):
        f1 = self.global_encoder(x)

        f2 = self.local_projector(f1)

        local_c = self.local_C(f2)
        global_c = self.global_C(f1)

        return f2, f1, local_c, global_c

def BuildModel(model_name, num_classes, device):
    return ResNet(model_type=model_name, num_classes=num_classes).to(device)
