import torch
import torch.nn as nn
from einops import rearrange


class DeepONet1D(nn.Module):
    def __init__(self, in_channels, out_dims=1, init_channels=32, mid_dim=150) -> None:
        super().__init__()
        self.act = nn.Tanh()
        self.shape_last_fm = 32
        self.mid_dim = mid_dim

        self.init_layer = nn.Sequential(nn.Conv1d(in_channels, init_channels, kernel_size=3), self.act, nn.AvgPool1d(kernel_size=2))

        self.cnn_b_layer1 = nn.Sequential(nn.Conv1d(init_channels, init_channels, kernel_size=3), self.act, nn.AvgPool1d(kernel_size=2))
        self.cnn_b_layer2 = nn.Sequential(nn.Conv1d(init_channels, init_channels, kernel_size=3), self.act, nn.AvgPool1d(kernel_size=2))
        self.cnn_b_layer3 = nn.Sequential(nn.Conv1d(init_channels, init_channels, kernel_size=3), self.act, nn.AdaptiveAvgPool1d(self.shape_last_fm))

        self.fnn_b_layer1 = nn.Sequential(nn.Linear(init_channels*self.shape_last_fm, 512), self.act)
        self.fnn_b_layer2 = nn.Sequential(nn.Linear(512, 256), self.act)
        self.fnn_b_layer3 = nn.Sequential(nn.Linear(256, mid_dim*out_dims), self.act)

        self.fnn_t_layer1 = nn.Sequential(nn.Linear(1, 128), self.act)
        self.fnn_t_layer2 = nn.Sequential(nn.Linear(128, 128), self.act)
        self.fnn_t_layer3 = nn.Sequential(nn.Linear(128, 128), self.act)
        self.fnn_t_layer4 = nn.Sequential(nn.Linear(128, mid_dim*out_dims), self.act)

    def forward(self, u, x):
        u = rearrange(u, 'b h c -> b c h')
        x = rearrange(x, 'b h c -> b c h')
        # u: [b c h w], x: [b 2 h w]
        b = self.branch_net(u)
        b = rearrange(b, 'b (c1 c2) -> b c1 c2', c2=self.mid_dim)
        t = self.trunck_net(rearrange(x, 'b c h -> b h c'))
        t = rearrange(t, 'b h (c1 c2) -> b h c1 c2', c2=self.mid_dim)
        h = (b[:, None]*t).sum(-1, keepdim=False)
        # return rearrange(h, 'b h w c -> b c h w')
        return h.unsqueeze(-2)
    
    def branch_net(self, u):
        b = self.init_layer(u)
        b = self.cnn_b_layer1(b)
        b = self.cnn_b_layer2(b)
        b = self.cnn_b_layer3(b)
        b = self.fnn_b_layer1(rearrange(b, 'b c h -> b (c h)'))
        b = self.fnn_b_layer2(b)
        b = self.fnn_b_layer3(b)
        return b
    
    def trunck_net(self, x):
        t = self.fnn_t_layer1(x)
        t = self.fnn_t_layer2(t)
        t = self.fnn_t_layer3(t)
        t = self.fnn_t_layer4(t)
        return t


class DeepONet2D(nn.Module):
    def __init__(self, in_channels, out_dims=1, init_channels=32, mid_dim=150) -> None:
        super().__init__()
        self.act = nn.Tanh()
        self.shape_last_fm = (16, 16)
        self.mid_dim = mid_dim

        self.init_layer = nn.Sequential(nn.Conv2d(in_channels, init_channels, kernel_size=3), self.act, nn.AvgPool2d(kernel_size=2))

        self.cnn_b_layer1 = nn.Sequential(nn.Conv2d(init_channels, init_channels, kernel_size=3), self.act, nn.AvgPool2d(kernel_size=2))
        self.cnn_b_layer2 = nn.Sequential(nn.Conv2d(init_channels, init_channels, kernel_size=3), self.act, nn.AvgPool2d(kernel_size=2))
        self.cnn_b_layer3 = nn.Sequential(nn.Conv2d(init_channels, init_channels, kernel_size=3), self.act, nn.AdaptiveAvgPool2d(self.shape_last_fm))

        self.fnn_b_layer1 = nn.Sequential(nn.Linear(init_channels*self.shape_last_fm[0]*self.shape_last_fm[1], 512), self.act)
        self.fnn_b_layer2 = nn.Sequential(nn.Linear(512, 256), self.act)
        self.fnn_b_layer3 = nn.Sequential(nn.Linear(256, mid_dim*out_dims), self.act)

        self.fnn_t_layer1 = nn.Sequential(nn.Linear(2, 128), self.act)
        self.fnn_t_layer2 = nn.Sequential(nn.Linear(128, 128), self.act)
        self.fnn_t_layer3 = nn.Sequential(nn.Linear(128, 128), self.act)
        self.fnn_t_layer4 = nn.Sequential(nn.Linear(128, mid_dim*out_dims), self.act)

    def forward(self, u, x):
        u = rearrange(u, 'b h w c -> b c h w')
        x = rearrange(x, 'b h w c -> b c h w')
        # u: [b c h w], x: [b 2 h w]
        b = self.branch_net(u)
        b = rearrange(b, 'b (c1 c2) -> b c1 c2', c2=self.mid_dim)
        t = self.trunck_net(rearrange(x, 'b c h w -> b h w c'))
        t = rearrange(t, 'b h w (c1 c2) -> b h w c1 c2', c2=self.mid_dim)
        h = (b[:, None, None]*t).sum(-1, keepdim=False)
        # return rearrange(h, 'b h w c -> b c h w')
        return h.unsqueeze(-2)
    
    def branch_net(self, u):
        b = self.init_layer(u)
        b = self.cnn_b_layer1(b)
        b = self.cnn_b_layer2(b)
        b = self.cnn_b_layer3(b)
        b = self.fnn_b_layer1(rearrange(b, 'b c h w -> b (c h w)'))
        b = self.fnn_b_layer2(b)
        b = self.fnn_b_layer3(b)
        return b
    
    def trunck_net(self, x):
        t = self.fnn_t_layer1(x)
        t = self.fnn_t_layer2(t)
        t = self.fnn_t_layer3(t)
        t = self.fnn_t_layer4(t)
        return t


class MIONet2D(nn.Module):
    def __init__(self, in_channels, out_dims=1, init_channels=32, mid_dim=150) -> None:
        super().__init__()
        self.act = nn.Tanh()
        self.shape_last_fm = (8, 8)
        self.mid_dim = mid_dim
        self.num_fn = in_channels

        # self.init_layer = nn.Sequential(nn.Conv2d(in_channels, init_channels, kernel_size=3), self.act, nn.AvgPool2d(kernel_size=2))

        branches = nn.ModuleList([])
        for i in range(self.num_fn):
            branches.append(nn.Sequential(
                nn.Conv2d(1, init_channels, kernel_size=3), self.act, nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(init_channels, init_channels, kernel_size=3), self.act, nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(init_channels, init_channels, kernel_size=3), self.act, nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(init_channels, init_channels, kernel_size=3), self.act, nn.AdaptiveAvgPool2d(self.shape_last_fm)
            ))
            branches.append(nn.Sequential(
                nn.Linear(init_channels*self.shape_last_fm[0]*self.shape_last_fm[1], 256), self.act,
                nn.Linear(256, mid_dim*out_dims), self.act
            ))
        self.branches = branches

        self.fnn_t_layer1 = nn.Sequential(nn.Linear(2, 128), self.act)
        self.fnn_t_layer2 = nn.Sequential(nn.Linear(128, 128), self.act)
        self.fnn_t_layer3 = nn.Sequential(nn.Linear(128, 128), self.act)
        self.fnn_t_layer4 = nn.Sequential(nn.Linear(128, mid_dim*out_dims), self.act)

    def forward(self, u, x):
        u = rearrange(u, 'b h w c -> b c h w')
        x = rearrange(x, 'b h w c -> b c h w')
        # u: [b c h w], x: [b 2 h w]
        b = self.branch_net(u)
        b = rearrange(b, 'b (c1 c2) -> b c1 c2', c2=self.mid_dim)
        t = self.trunck_net(rearrange(x, 'b c h w -> b h w c'))
        t = rearrange(t, 'b h w (c1 c2) -> b h w c1 c2', c2=self.mid_dim)
        h = (b[:, None, None]*t).sum(-1, keepdim=False)
        # return rearrange(h, 'b h w c -> b c h w')
        return h.unsqueeze(-2)
    
    def branch_net(self, u):
        ilayer = 0
        for i in range(self.num_fn):
            b = self.branches[ilayer](u[:, [i]])
            ilayer += 1
            b = rearrange(b, 'b c h w -> b (c h w)')
            b = self.branches[ilayer](b)
            ilayer += 1
            if i==0:
                feats = b
            else:
                feats = feats * b
        return feats
    
    def trunck_net(self, x):
        t = self.fnn_t_layer1(x)
        t = self.fnn_t_layer2(t)
        t = self.fnn_t_layer3(t)
        t = self.fnn_t_layer4(t)
        return t
