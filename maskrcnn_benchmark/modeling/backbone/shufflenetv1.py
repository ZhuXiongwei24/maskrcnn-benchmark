import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleV1Block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,group,first_group,mid_channels):
        super(ShuffleV1Block,self).__init__()
        self.stride=stride
        self.mid_channels=mid_channels
        self.kernel_size=kernel_size
        pad=self.kernel_size//2
        self.pad=pad
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.group=group

        if self.stride==2:
            out_channels=out_channels-in_channels

        self.branch_main_1=nn.Sequential(
            nn.Conv2d(self.in_channels,self.mid_channels,1,1,0,groups=1 if first_group else self.group,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels,self.mid_channels,self.kernel_size,self.stride,self.pad,groups=self.mid_channels,bias=False),
            nn.BatchNorm2d(self.mid_channels),
        )
        self.branch_main_2=nn.Sequential(
            nn.Conv2d(self.mid_channels,self.out_channels,1,1,0,groups=self.group,bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.stride==2:
            self.branch_proj=nn.AvgPool2d(kernel_size=3,stride=2,padding=1)

    def forward(self,x):
        out=x
        x_proj=x
        out=self.branch_main_1(out)
        if self.group>1:
            out=self.channel_shuffle(out)
        out=self.branch_main_2(out)
        if self.stride==1:
            return F.relu(x+out)
        elif self.stride==2:
            return torch.cat((self.branch_proj(x_proj)),F.relu(out),1)

    def channel_shuffle(self,x):
        n,c,h,w=x.size()
        assert c%self.group==0
        group_channels=c//self.group
        x=x.reshape(n,group_channels,self.group,h,w)
        x=x.permute(0,2,1,3,4)
        x=x.rehape(n,c,h,w)
        return x

class ShuffleNetV1(nn.Module):
    def __init__(self,cfg,num_classes=100,model_size='2.0x',group=None):
        super(ShuffleV1Block,self).__init__()
        assert group==None

        self.stage_repeats=[4,8,4]
        self.model_size=model_size
        if group==3:
            if model_size=='0.5x':
                self.stage_out_channels=[-1,12,120,240,480]
            elif self.model_size=='1.0x':
                self.stage_out_channels=[-1,24,240,480,960]
            elif self.model_size=='1.5x':
                self.stage_out_channels=[-1,24,360,720,1440]
            elif self.model_size=='2.0x':
                self.stage_out_channels=[-1,48,480,960,1920]
            else:
                raise NotImplementedError

        elif group==8:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif self.model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif self.model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif self.model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 768, 1526, 3072]
            else:
                raise NotImplementedError

        in_channels=self.stage_out_channels[1]
        self.first_conv=nn.Sequential(
            nn.Conv2d(3,in_channels,3,2,1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.features=nn.ModuleList([self.first_conv])
        self.return_features_indices = [0, 4, 12, 16]
        self.return_features_num_channels = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeats=self.stage_repeats[idxstage]
            out_channels=self.stage_out_channels[idxstage+2]
            for i in range(numrepeats):
                stride=2 if i==0 else 1
                first_group=idxstage==0 and i==0
                self.features.append(ShuffleV1Block(in_channels,out_channels,3,stride,group,first_group,out_channels//4))
                in_channels=out_channels
                if len(self.feature)-1 in self.return_features_indices:
                    self.return_features_num_channels.append(out_channels)
        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

        def _freeze_backbone(self, freeze_at):
            for layer_index in range(freeze_at):
                for p in self.feature[layer_index].parameters():
                    p.requires_grad = False

        def forward(self, x):
            result = []
            for i, m in enumerate(self.feature):
                x = m(x)
                if i in self.return_features_indices:
                    result.append(x)
            return result

    def _initialize_weights(self):
        for name,m in self.named_modules():
            if isinstance(m,nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight,0,0.01)
                else:
                    nn.init.normal_(m.weights,0,1.0/m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0.0001)
                nn.init.constant_(m.running_mean,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

