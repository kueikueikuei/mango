import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet

class DFL_VGG16(nn.Module):
    def __init__(self, k = 10, nclass = 200):
        super(DFL_VGG16, self).__init__()
        self.k = k
        self.nclass = nclass

        # k channels for one class, nclass is total classes, therefore k * nclass for conv6
        vgg16featuremap = torchvision.models.vgg16_bn(pretrained=True).features
        conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-11])
        conv5 = torch.nn.Sequential(*list(vgg16featuremap.children())[-11:])
        conv6 = torch.nn.Conv2d(512, k * nclass, kernel_size = 1, stride = 1, padding = 0)
        pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

        # Feature extraction root
        self.conv1_conv4 = conv1_conv4

        # G-Stream
        self.conv5 = conv5
        self.cls5 = nn.Sequential(
            nn.Conv2d(512, nclass, kernel_size=1, stride = 1, padding = 0),
            nn.BatchNorm2d(nclass),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            )

        # P-Stream
        self.conv6 = conv6
        self.pool6 = pool6
        self.cls6 = nn.Sequential(
            nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
            nn.AdaptiveAvgPool2d((1,1)),
            )

        # Side-branch
        self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)

    def forward(self, x):
        batchsize = x.size(0)

        # Stem: Feature extraction
        inter4 = self.conv1_conv4(x)

        # G-stream
        x_g = self.conv5(inter4)
        out1 = self.cls5(x_g)
        out1 = out1.view(batchsize, -1)

        # P-stream ,indices is for visualization
        x_p = self.conv6(inter4)
        x_p, indices = self.pool6(x_p)
        inter6 = x_p
        out2 = self.cls6(x_p)
        out2 = out2.view(batchsize, -1)

        # Side-branch
        inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
        out3 = self.cross_channel_pool(inter6)
        out3 = out3.view(batchsize, -1)

        return out1, out2, out3, x_g, x_p
    
class DFL_RESNET(nn.Module):
    def __init__(self, k = 200, nclass = 3):
        super(DFL_RESNET, self).__init__()
        self.k = k
        self.nclass = nclass

        # k channels for one class, nclass is total classes, therefore k * nclass for conv6
        conv1_conv4 = torch.nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-3])
        conv5 = torch.nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[-3:-2])
        conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
        pool6 = torch.nn.MaxPool2d((14, 14), stride = (14, 14), return_indices = True)

        # Feature extraction root
        self.conv1_conv4 = conv1_conv4

        # G-Stream
        self.conv5 = conv5
        self.cls5 = nn.Sequential(
            nn.Conv2d(2048, nclass, kernel_size=1, stride = 1, padding = 0),
            nn.BatchNorm2d(nclass),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            )

        # P-Stream
        self.conv6 = conv6
        self.pool6 = pool6
        self.cls6 = nn.Sequential(
            nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
            nn.AdaptiveAvgPool2d((1,1)),
            )

        # Side-branch
        self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)

    def forward(self, x):
        batchsize = x.size(0)

        # Stem: Feature extraction
        inter4 = self.conv1_conv4(x)

        # G-stream
        x_g = self.conv5(inter4)
        out1 = self.cls5(x_g)
        out1 = out1.view(batchsize, -1)

        # P-stream ,indices is for visualization
        x_p = self.conv6(inter4)
        x_p, indices = self.pool6(x_p)
        inter6 = x_p
        out2 = self.cls6(x_p)
        out2 = out2.view(batchsize, -1)

        # Side-branch
        inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
        out3 = self.cross_channel_pool(inter6)
        out3 = out3.view(batchsize, -1)

        return out1, out2, out3


    
class DFL_EfficientNet(nn.Module):
    def __init__(self, k = 200, nclass = 3):
        super(DFL_EfficientNet, self).__init__()
        self.k = k
        self.nclass = nclass
        self.backbone = EfficientNet.from_pretrained('efficientnet-b7', num_classes=3)

        # k channels for one class, nclass is total classes, therefore k * nclass for conv6
        self.stem=torch.nn.Sequential(*list(self.backbone.children())[0:2])
        conv1_conv4 = torch.nn.Sequential(*list(list(self.backbone.children())[2].children())[0:54])
        conv5 = torch.nn.Sequential(*list(list(self.backbone.children())[2].children())[54:55])
        conv6 = torch.nn.Conv2d(640, k * nclass, kernel_size = 1, stride = 1, padding = 0)
        pool6 = torch.nn.MaxPool2d((7, 7), stride = (7, 7), return_indices = True)

        # Feature extraction root
        self.conv1_conv4 = conv1_conv4

        # G-Stream
        self.conv5 = conv5
        self.cls5 = nn.Sequential(
            nn.Conv2d(640, nclass, kernel_size=1, stride = 1, padding = 0),
            nn.BatchNorm2d(nclass),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            )

        # P-Stream
        self.conv6 = conv6
        self.pool6 = pool6
        self.cls6 = nn.Sequential(
            nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
            nn.AdaptiveAvgPool2d((1,1)),
            )

        # Side-branch
        self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)

    def forward(self, x):
        batchsize = x.size(0)

        # Stem: Feature extraction
        x = self.stem(x)
        inter4 = self.conv1_conv4(x)

        # G-stream
        x_g = self.conv5(inter4)
        out1 = self.cls5(x_g)
        out1 = out1.view(batchsize, -1)

        # P-stream ,indices is for visualization
        x_p = self.conv6(inter4)
        x_p, indices = self.pool6(x_p)
        inter6 = x_p
        out2 = self.cls6(x_p)
        out2 = out2.view(batchsize, -1)

        # Side-branch
        inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
        out3 = self.cross_channel_pool(inter6)
        out3 = out3.view(batchsize, -1)

        return out1, out2, out3

class MangoNet(torch.nn.Module):
    def __init__(self):
        super(MangoNet, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b7', num_classes=3)
    def forward(self, x):
        x = self.backbone(x)
        return x
class MangoRegressionNet(torch.nn.Module):
    def __init__(self):
        super(MangoRegressionNet, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b7', num_classes=1)
    def forward(self, x):
        x = self.backbone(x)
        return x