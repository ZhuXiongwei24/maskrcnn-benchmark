import torch
import torch.nn as nn

class DepthWiseSeperabelConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(DepthWiseSeperabelConv2d,self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU
        )

    def forward(self,x):
        x=self.depthwise(x)
        x=self.pointwise(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MobileNetV1(nn.Module):
    def __init__(self,cfg,width_multiplier=1,num_classes=10):
        super(MobileNetV1,self).__init__()
        self.alpha=width_multiplier
        in_channels=64
        mobilenet_setting=[
            # c,n,s
            [128,2,2],
            [256,2,2],
            [512,6,2],
            [1024,2,2]
        ]
        in_channels=int(in_channels*self.alpha)
        self.return_features_indices=[2,4,10,12]
        self.return_feature_num_channels=[]
        self.features=nn.ModuleList([BasicConv2d(3,int(32*self.alpha),3,padding=1,bias=False),
                                     DepthWiseSeperabelConv2d(int(32*self.alpha),in_channels,32,padding=1,bias=False)])
        for c,n,s in mobilenet_setting:
            out_channels=int(self.alpha*c)
            for i in range(n):
                if i==0:
                    self.features.append(DepthWiseSeperabelConv2d(in_channels,out_channels,3,stride=s,padding=1,bias=False))
                else:
                    self.features.append(DepthWiseSeperabelConv2d(out_channels,out_channels,3,stride=1,padding=1,bias=False))
                in_channels=out_channels
                if len(self.features)-1 in self.return_features_indices:
                    self.return_feature_num_channels.append(out_channels)
        self._initialize_weights()
        self.freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self,freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_gead=False

    def forward(self,x):
        result=[]
        for i,m in enumerate(self.features):
            x=m(x)
            if i in self.return_features_indices:
                result.append(x)
        return result

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,(2./n)**0.5)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m,nn.Linear()):
                n=m.weight.size(1)
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
                