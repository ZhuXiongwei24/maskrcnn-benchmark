import torch
import torch.nn as nn

class NonLocalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 ratio,
                 use_scale=True,
                 mode='embeded_gaussian'):
        super(NonLocalBlock,self).__init__()
        self.in_channels=in_channels
        self.ratio=ratio
        self.use_scale=use_scale
        self.inner_channels=int(self.ratio*self.in_channels)
        assert mode in ['embedded_gaussian','dot_product','concatenation']
        self.conv_theta=nn.Conv2d(self.in_channels,self.inner_channels,1)
        self.conv_phi=nn.Conv2d(self.in_channels,self.inner_channels,1)
        self.conv_g=nn.Conv2d(self.in_channels,self.inner_channels,1)
        self.conv=nn.Conv2d(self.inner_channels,self.in_channels,1)

        self.weight_init()
    def weight_init(self,std=0.01,zero_init=True):
        for m in [self.conv_g,self.conv_theta,self.conv_phi,]:
            nn.init.normal_(m.weight,mean=0,std=std)
            if hasattr(m,'bias') and m.bias is not None:
                nn.init.constant_(m.bias,bias=0)
        if zero_init:
            if hasattr(self.conv,'weight') and self.conv.weight is not None:
                nn.init.constant_(self.conv.weight,0)
            if hasattr(self.conv,'bias') and self.conv.bias is not None:
                nn.init.constant_(self.conv.bias,0)
        else:
            nn.init.normal_(self.conv.weight,mean=0,std=std)
            if hasattr(self.conv,'bias') and self.conv.bias is not None:
                nn.init.constant_(self.conv.bias,0)

    def embeded_gaussian(self,theta_x,phi_x):
        # the shape of theta_x is [N,H*W,C], the shape of phi_x is [N,C,H*W]
        pairwise_weight=torch.matmul(theta_x,phi_x)
        # theta_x.shape[-1]=C=inner_channels
        if self.use_scale:
            pairwise_weight=pairwise_weight/theta_x.shape[-1]**0.5
        pairwise_weight=pairwise_weight.softmax(dim=-1)

        return pairwise_weight

    def dot_product(self,theta_x,phi_x):
        # the shape of theta_x is [N,H*W,C], the shape of phi_x is [N,C,H*W]
        pairwise_weight=torch.matmul(theta_x,phi_x)
        # theta_x.shape[-1]=C=inner_channels
        pairwise_weight=pairwise_weight/theta_x.shape[-1]

        return pairwise_weight

    def forward(self,x):
        n,_,h,w=x.shape
        # [N,H*W,C]
        g_x=self.conv_g(x).view(n,self.inner_channels,-1)
        g_x=g_x.permute(0,2,1)

        # [N,H*W,C]
        theta_x=self.conv_theta(x).view(n,self.inner_channels,-1)
        theta_x=theta_x.permute(0,2,1)

        # [N,C,H*W]
        phi_x=self.conv_phi(x).view(n,self.inner_channels,-1)

        pairwise_func=getattr(self.self.mode)

        # [N,H*W,H*W]
        pairwise_weight=pairwise_func(theta_x,phi_x)

        # [N,H*W,C]
        y=torch.matmul(pairwise_weight,g_x)

        y=y.permute(0,2,1).reshape(n,self.inner_channels,h,w)

        output=x+self.conv_phi(y)

        return output







