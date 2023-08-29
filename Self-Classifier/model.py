import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


class MLP(nn.Module):
    def __init__(self, in_dim=512, out_dim=128, hidden_dim=4096):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class Heads(nn.Module):
    def __init__(self, in_dim=128, sizes=(10, 20, 40, 80)):
        super().__init__()

        heads = []
        for size in sizes:
            head = nn.utils.weight_norm(nn.Linear(in_dim, size, bias=False))
            head.weight_g.data.fill_(1)
            heads.append(head)

        self.heads = nn.ModuleList(heads)

    def forward(self, x, train=True):

        output = [h(x) for h in self.heads] if train else self.heads[0](x)
        return output


class Model(nn.Module):
    # in_dim = feature_dim, out_dim = class_num
    def __init__(self, hidden_dim=(4096, 128),
                 head_sizes=(10, 20, 40, 80), backbone='resnet18'):

        super().__init__()
        if backbone == 'resnet50':
            backbone = resnet50()
            feature_dim = 2048
        elif backbone == 'resnet34':
            backbone = resnet34()
            feature_dim = 512
        elif backbone == 'resnet18':
            backbone = resnet18()
            feature_dim = 512

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.mlp = MLP(in_dim=feature_dim, hidden_dim=hidden_dim[0], out_dim=hidden_dim[1])
        self.heads = Heads(in_dim=hidden_dim[1], sizes=head_sizes)

    def forward(self, x, train=True):
        x = self.backbone(x)
        x = self.mlp(x)
        x = self.heads(x, train)
        return x


class LCModel(nn.Module):
    def __init__(self, in_model, num_classes=10):
        super().__init__()

        self.backbone = in_model.backbone
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False # Freeze params

        feature_dim = 512 # Match backbone
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(feature_dim, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        output = self.linear(x)

        return output


def transform_model(model, num_classes):
    model = nn.Sequential(*list(model.children())[:-2])

    for p in model.parameters():
        p.requires_grad = False

    model = nn.Sequential(model, nn.Linear(512, num_classes))
    return model