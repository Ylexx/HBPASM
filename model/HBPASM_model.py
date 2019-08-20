import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import resnet_model

def mask_Generation(feature, alpha):
    batch_size = feature.size(0)
    kernel = feature.size(2)
    sum = torch.sum(feature.detach(), dim=1)

    avg = torch.sum(torch.sum(sum, dim=1), dim=1) / kernel ** 2

    mask = torch.where(sum > alpha * avg.view(batch_size, 1, 1), torch.ones(sum.size()).cuda(),
                       (torch.zeros(sum.size()) + 0.1).cuda())

    mask = mask.unsqueeze(1)
    return mask

class Net(nn.Module):
    def __init__(self, model_path):
        super(Net, self).__init__()

        self.proj0 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)
        self.proj1 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)
        self.proj2 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)

        # fc layer
        self.fc_concat = torch.nn.Linear(8192 * 3, 200)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, 0.01)
        #         m.bias.data.zero_()

        self.softmax = nn.LogSoftmax(dim=1)
        self.avgpool = nn.AvgPool2d(kernel_size=14)

        # base-model
        self.features = resnet_model.resnet34(pretrained=True,
                                              model_root=model_path)

    def forward(self, x):
        batch_size = x.size(0)
        feature4_0, feature4_1, feature4_2 = self.features(x)

        slack_mask1 = mask_Generation(feature4_0, alpha=0.6)
        slack_mask2 = mask_Generation(feature4_1, alpha=0.6)
        slack_mask3 = mask_Generation(feature4_2, alpha=0.6)

        Aggregated_mask = slack_mask1 * slack_mask2 * slack_mask3

        feature4_0 = feature4_0 * Aggregated_mask
        feature4_1 = feature4_1 * Aggregated_mask
        feature4_2 = feature4_2 * Aggregated_mask

        feature4_0 = self.proj0(feature4_0)
        feature4_1 = self.proj1(feature4_1)
        feature4_2 = self.proj2(feature4_2)

        inter1 = feature4_0 * feature4_1
        inter2 = feature4_0 * feature4_2
        inter3 = feature4_1 * feature4_2

        inter1 = self.avgpool(inter1).view(batch_size, -1)
        inter2 = self.avgpool(inter2).view(batch_size, -1)
        inter3 = self.avgpool(inter3).view(batch_size, -1)

        result1 = torch.nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
        result2 = torch.nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
        result3 = torch.nn.functional.normalize(torch.sign(inter3) * torch.sqrt(torch.abs(inter3) + 1e-10))

        result = torch.cat((result1, result2, result3), 1)
        result = self.fc_concat(result)
        return self.softmax(result)
