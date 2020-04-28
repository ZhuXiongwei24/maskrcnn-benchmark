import torch
import torch.nn as nn

def BasicConv2d(in_channels,out_channels,stride):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,stride,1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self,in_channels,out_channels,stride,expand_ratio):
        super(InvertedResidual,self).__init__()
        self.stride=stride
        hidden_dim=int(round(in_channels*expand_ratio))
        self.use_res_connections=self.stride==1 and in_channels==out_channels
        if expand_ratio==1:
            self.conv=nn.Sequential(
                nn.Conv2d(hidden_dim,hidden_dim,3,stride,1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim,out_channels,1,1,0,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv=nn.Sequential(
                nn.Conv2d(in_channels,hidden_dim,1,1,0,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim,hidden_dim,3,stride,1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim,out_channels,1,1,0,bias=False),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self,x):
        if self.use_res_connections:
            return x+self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,cfg,width_multiplier=1,num_classes=10):
        super(MobileNetV2,self).__init__()
        block=InvertedResidual
        in_channels=32
        interverted_residual_setting=[
            # t,c,n,s
            [1,16,1,1],
            [6,24,2,2],
            [6,32,3,2],
            [6,64,4,2],
            [6,96,3,1],
            [6,160,3,2],
            [6,320,1,1],
        ]

        in_channels=int(in_channels*width_multiplier)
        self.return_features_indices=[2,6,13,17]
        self.return_features_num_channels=[]
        self.feature=nn.ModuleList([BasicConv2d(3,in_channels,2)])
        for t,c,n,s in interverted_residual_setting:
            out_channels=int(c*width_multiplier)
            for i in range(n):
                if i==0:
                    self.feature.append(block(in_channels,out_channels,s,expand_ratio=t))
                else:
                    self.features.append(block(in_channels,out_channels,1,expand_ratio=t))
                in_channels=out_channels
                if len(self.feature)-1 in self.return_features_indices:
                    self.return_features_num_channels.append(out_channels)
        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)
    def _freeze_backbone(self,freeze_at):
        for layer_index in range(freeze_at):
            for p in self.feature[layer_index].parameters():
                p.requires_grad=False

    def forward(self,x):
        result=[]
        for i,m in enumerate(self.feature):
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
            elif isinstance(m,nn.Linear):
                n=m.weight.size(1)
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()