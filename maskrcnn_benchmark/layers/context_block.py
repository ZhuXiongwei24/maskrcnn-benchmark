import torch
import torch.nn as nn
def const_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['normal', 'uniform']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)

    else:
        nn.init_kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)

    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        const_init(m[-1], 0)
    else:
        const_init(m,val=0)


class ContextBlock(nn.Module):
    def __init__(self,
                 inplaces,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock,self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types,(list, tuple))
        valid_fusion_types=['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types)>0, 'at least one fusion type should bo used.'
        self.inplaces=inplaces
        self.ratio=ratio
        self.planes=int(inplaces*ratio)
        self.pooling_type=pooling_type
        self.fusion_types=fusion_types
        if pooling_type=='att':
            self.conv_mask=nn.Conv2d(inplaces,1,kernel_size=1)
            self.softmax=nn.SoftMax(dim=2)

        else:
            self.avg_pool=nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv=nn.Sequential(
                nn.Conv2d(self.inplaces,self.planes,kernel_size=1),
                nn.LayerNorm([self.planes,1,1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes,self.inplaces,kernel_size=1)
            )
        else:
            self.channel_add_conv=None

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv=nn.Sequential(
                nn.Conv2d(self.inplaces,self.planes,kernel_size=1),
                nn.LayerNorm([self.planes,1,1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes,self.inplaces,kernel_size=1)
            )
        else:
            self.channel_mul_conv=None
        self.resnet_parameters()

    def resnet_parameters(self):
        if self.pooling_type=='att':
            kaiming_init(self.conv_mask,mode='fan_in')
            self.conv_mask.inited=True
        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self,x):
        batch, channel,height,width=x.size()
        if self.pooling_type=='att':
            input_x=x
            #[N,C,H*W]
            input_x=input_x.view(batch,channel,height*width)
            #[N,1,C,H*W]
            input_x=input_x.unsqueeze(1)
            #[N,1,H,W]
            context_mask=self.conv_mask(x)
            #[N,1,H*W]
            context_mask=context_mask.view(batch,1,height*width)
            #[N,1,H*W,1]
            context_mask=context_mask.unsqueeze(-1)
            #[N,1,C,1]
            context=torch.matmul(input_x,context_mask)
            #[N,C,1,1]
            context=context.view(batch,channel,1,1)
        else:
            context=self.avg_pool(x)
        return context

    def forward(self,x):
        context=self.spatial_pool(x)
        out=x
        if self.channel_mul_conv is not None:
            channel_mul_term=torch.sigmoid(self.channel_mul_conv(context))
            out=out*channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term=torch.sigmoid(self.channel_add_conv(context))
            out=out+channel_add_term

        return out
