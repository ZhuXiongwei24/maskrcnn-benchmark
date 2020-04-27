import torch
import torch.nn

class SEBlock(nn.Module):
    def __init__(self,
                 inplaces,
                 ratio):
        super(SEBlock,self).__init__()
        self.inplaces=inplaces
        self.ratio=ratio
        self.places=int(inplaces*ratio)
        self.avg_pool=nn.AdaptivePool2d(1)
        self.conv=nn.Sequential(
            nn.Conv2d(self.inplaces,self.places,kernel_size=1),
            nn.BatchNorm2d(self.planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.places,kernel_size=1)
        )
        for modules in [self.conv,]:
            for l in modules.modules():
                if isinstance(l,nn.Conv2d):
                    nn.init.kaiming_uniform(l.weight,a=1,mode='fan_in')
                    if hasattr(l,'bias') or l.bias is not None:
                        nn.init.constant_(l.bias,0)


    def forward(self,x):
        out=x
        context=self.avg_pool(x)
        channel_mul_term=torch.sigmoid(self.conv(context))
        out=out*channel_mul_term

        return out