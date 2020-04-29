import torch
import torch.nn as nn
import torch.nn.functional as F

class hsigmoid(nn.Module):
    def forward(self,x):
        output=F.relu6(x+6,inplace=True)/6
        return output

class hswish(nn.Module):
    def forward(self,x):
        output=x*F.relu6(x+6,inplace=True)/6
        return output

class SeBlock(nn.Module):
    def __init__(self,inplaces,se_ratio=0.25):
        super(SeBlock,self).__init__()
        self.se_block=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplaces,int(inplaces*se_ratio),kernel_size=1,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(int(inplaces*se_ratio)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(inplaces*se_ratio),inplaces,kernel_size=1,stride=1,padding=1,bias=False),
            hsigmoid(),
        )
    def forward(self,x):
        return x*self.se_block(x)

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,expand_size,nonlinear,se_block):
        super(Block,self).__init__()
        self.se=se_block
        self.conv1=nn.Conv2d(in_channels,expand_size,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=nn.BatchNorm2d(expand_size)
        self.nonlinear1=nonlinear
        self.conv2=nn.Conv2d(expand_size,expand_size,kernel_size=kernel_size,stride=stride,padding=kernel_size//2,groups=expand_size,bias=False)
        self.bn2=nn.BatchNorm2d(expand_size)
        self.nonlinear2=nonlinear
        self.conv3=nn.Conv2d(expand_size,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)

        self.shortcut=nn.Sequential()
        if stride==1 and in_channels!=out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self,x):
        output=self.nonlinear1(self.bn1(self.conv1(x)))
        output=self.nonlinear2(self.bn2(self.conv2(output)))
        output=self.bn3(self.conv3(output))
        if self.se!=None:
            output=self.se(output)

        output+=self.shortcut(x) if self.stride==1 else output
        return output

class MobileNetV3_Large(nn.Module):
    def __init__(self,cfg,num_classes=100):
        super(MobileNetV3_Large,self).__init__()
        self.return_features_indices = [3, 6, 12, 16]
        self.return_features_num_channels = [24, 40, 112, 160]
        self.features=nn.ModuleList()
        self.stem=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(16),
            hswish(),
        )
        self.features.append(self.stem)
        self.features.append(Block(16,16,3,1,16,nn.ReLU(inplace=True),None))
        self.features.append(Block(16,24,3,2,64,nn.ReLU(inplace=True),None))
        self.features.append(Block(24,24,3,1,72,nn.ReLU(inplace=True),None))
        self.features.append(Block(24,40,5,2,72,nn.ReLU(inplace=True),SeBlock(40)))
        self.features.append(Block(40,40,5,1,120,nn.ReLU(inplace=True),SeBlock(40)))
        self.features.append(Block(40,40,5,1,120,nn.ReLU(inplace=True),SeBlock(40)))
        self.features.append(Block(40,80,3,2,240,hswish(),None))
        self.features.append(Block(80,80,3,1,200,hswish(),None))
        self.features.append(Block(80,80,3,1,184,hswish(),None))
        self.features.append(Block(80,80,3,1,184,hswish(),None))
        self.features.append(Block(80,112,3,1,480,hswish(),SeBlock(112)))
        self.features.append(Block(112,112,3,1,672,hswish(),SeBlock(112)))
        self.features.append(Block(112,160,5,2,672,hswish(),SeBlock(160)))
        self.features.append(Block(160,160,5,1,960,hswish(),SeBlock(160)))
        self.features.append(Block(160, 160, 5, 1, 960, hswish(), SeBlock(160)))
        self.last_feature=nn.Sequential(
            nn.Conv2d(160,960,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(960),
            hswish(),
        )
        self._init_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.feature[layer_index].parameters():
                p.requires_grad = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self,x):
        result = []
        for i, m in enumerate(self.feature):
            x = m(x)
            if i in self.return_features_indices:
                result.append(x)
        return result

class MobileNetV3_Small(nn.Module):
    def __init__(self,cfg,num_classes=100):
        super(MobileNetV3_Small,self).__init__()
        self.return_features_indices = [3, 6, 8, 11]
        self.return_features_num_channels = [24, 40, 48, 96]
        self.features = nn.ModuleList()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish(),
        )
        self.features.append(self.stem)
        self.features.append(Block(16,16,3,2,16,nn.ReLU(inplace=True),SeBlock(16)))
        self.features.append(Block(16,24,3,2,72,nn.ReLU(inplace=True),None))
        self.features.append(Block(24,24,3,1,88,nn.ReLU(inplace=True),None))
        self.features.append(Block(24,40,5,2,96,hswish(),SeBlock(40)))
        self.features.append(Block(40,40,5,1,240,hswish(),SeBlock(40)))
        self.features.append(Block(40, 40, 5, 1, 240, hswish(), SeBlock(40)))
        self.features.append(Block(40, 48, 5, 1, 120, hswish(), SeBlock(48)))
        self.features.append(Block(48, 48, 5, 1, 144, hswish(), SeBlock(48)))
        self.features.append(Block(48, 96, 5, 2, 288, hswish(), SeBlock(96)))
        self.features.append(Block(96, 96, 5, 1, 576, hswish(), SeBlock(96)))
        self.features.append(Block(96, 96, 5, 1, 576, hswish(), SeBlock(96)))
        self.last_feature=nn.Sequential(
            nn.Conv2d(96,576,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(576),
            hswish(),
        )
        self._init_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.feature[layer_index].parameters():
                p.requires_grad = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        result = []
        for i, m in enumerate(self.feature):
            x = m(x)
            if i in self.return_features_indices:
                result.append(x)
        return result
