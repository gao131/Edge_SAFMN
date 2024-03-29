import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


# Channel MLP: Conv1*1 -> Conv1*1
class ChannelMLP(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mlp(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mbconv(x)


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim+12, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            # nn.PReLU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x,x0):
        return self.ccm(torch.cat([x,x0], dim=1))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        # self.relu1 = nn.PReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        
        # Activation
        self.act = nn.GELU()
        
        # self.channel_atten = ChannelAttention(chunk_dim)

        # self.cons = nn.ModuleList([nn.Conv2d(18, 9, 1, 1, 0) for i in range(self.n_levels-1)])

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s1 = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s1)
                
                # channel_atten = self.channel_atten(s)
                # s = s*channel_atten + s1
                
                s = F.interpolate(s, size=(h, w), mode='nearest')               
                # s = self.cons[i-1](torch.cat([s,xc[i]], dim=1))
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        # out = self.act(out)
        return out

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        # Multiscale Block
        self.safm = SAFM(dim) 
        
        # self.channel_atten = ChannelAttention(dim)
        # self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # self.act = nn.PReLU()
        # self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale) 

    def forward(self, x,x0):
        x = self.safm(self.norm1(x)) + x
        
        # x = self.safm(self.norm1(x)) + x
        # x1 = self.act(self.conv(x))
        # x1 = self.conv1(x1)
        # x = self.channel_atten(x1)*x1 + x 
        
        x = self.ccm(self.norm2(x),x0) + x
        return x



class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # self.num = 2
        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
            self.num = nn.Parameter(2*torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)
            self.num = nn.Parameter(2*torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)
            

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight1 = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)
        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        self.kernel_mid = kernel_mid
        self.out_channels = out_channels
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight1[idx, :, 0, self.kernel_mid] = -1
                self.sobel_weight1[idx, :, -1, self.kernel_mid] = 1
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, -1, :] = 1
            elif idx % 4 == 1:
                self.sobel_weight1[idx, :, self.kernel_mid, 0] = -1
                self.sobel_weight1[idx, :, self.kernel_mid, -1] = 1
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, :, -1] = 1
            elif idx % 4 == 2:
                self.sobel_weight1[idx, :, 0, 0] = -1
                self.sobel_weight1[idx, :, -1, -1] = 1
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
            else:
                self.sobel_weight1[idx, :, -1, 0] = -1
                self.sobel_weight1[idx, :, 0, -1] = 1
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
    

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = (self.sobel_weight + self.sobel_weight1*self.num) * self.sobel_factor
        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out    
    
@ARCH_REGISTRY.register()
class SAFMN(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        

        self.upscaling_factor = upscaling_factor
        
        self.feats = nn.ModuleList([AttBlock(dim, ffn_scale) for i in range(n_blocks)])
        self.n_blocks = n_blocks
        
        # 8X5SAFMN
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3*upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )
        self.sobelconv = SobelConv2d(3,12,padding=1)
        
        # self.cons1 = nn.Sequential(
        #     nn.Conv2d(dim+12, dim+12, 3, 1, 1),
        #     # nn.GELU(), 
        #     nn.PReLU(),
        #     # nn.Conv2d(dim, dim, 1, 1, 0)
        # )
        
        
        self.con1 = nn.Conv2d(6,9, 3, 1, 1)
        self.act = nn.PReLU()
        self.con2 = nn.Conv2d(9, 3, 3, 1, 1)
        
    def forward(self, x):
        x0 = self.sobelconv(x)
        
        h, w = x.size()[-2:]
        x1 = F.interpolate(x, size=(self.upscaling_factor*h,self.upscaling_factor*w), mode='bicubic',align_corners=False)
        
      
        
        x = self.to_feat(x)
        x2 = x
        
        for i in range(self.n_blocks):
            x = self.feats[i](x,x0)
        x = x + x2
        # x = self.cons1(torch.cat([x,x0], dim=1))
      
        
        x = self.to_img(x)

        
        x = self.con1(torch.cat([x,x1], dim=1))
        x = self.con2(self.act(x))
        
        
        
        return x



if __name__== '__main__':
    #############Test Model Complexity #############
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 3, 100, 100)
    # x = torch.randn(1, 3, 256, 256)

    model = SAFMN(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=4)
    # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    # output = model(x)
    # print(output.shape)
