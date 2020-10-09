import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):        
        super(Mish, self).__init__()
        
    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

    
class Activation(nn.Module):
    def __init__(self, activation: str):
        super(Activation, self).__init__()
        
        if activation == 'ReLU':        
            self.activation = nn.ReLU()
            
        elif activation == 'LReLU':
            self.activation = nn.LeakyReLU()
            
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
            
        elif activation == 'Linear':
            self.activation = nn.Identity()
            
        elif activation == 'Mish':
            self.activation = Mish()
            
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()

        elif activation == 'CELU':
            self.activation = nn.CELU()
            
        else:
            raise NotImplementedError("Not expected activation: %s"%activation)
            
    def forward(self, x):
        return self.activation(x)

    
class SPP(nn.Module):
    # Convolutional SPP network
    # Reference: https://github.com/WongKinYiu/PyTorch_YOLOv4
    def __init__(self, ch=128, kernel_sizes=[5, 9, 13], stride=1):
        super(SPP, self).__init__()
        _ch = ch //2
        # convolution layers to deal with increased channels
        self.conv1 = nn.Conv2d(ch, _ch, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(_ch*(len(kernel_sizes)+1), ch, 1, 1, bias=False)

        self.pooling_layers = nn.ModuleList()
        for kernel_size in kernel_sizes: 
            self.pooling_layers.append(nn.MaxPool2d(kernel_size, stride, (kernel_size-1)//2))
            
    def forward(self, x):
        x = self.conv1(x)
        y = [x]
        for pooling_layer in self.pooling_layers:
            y.append(pooling_layer(x))
        return self.conv2(torch.cat(y, dim=1))


class Pool(nn.Module):
    def __init__(self, channel: int, pool: str):
        super(Pool, self).__init__()
        if pool == 'Max':
            self.pool = nn.MaxPool2d(2, 2)
            
        elif pool == 'Avg':
            self.pool = nn.AvgPool2d(2, 2)
            
        elif pool == 'Conv':
            self.pool = nn.Conv2d(channel, channel, kernel_size=2, stride=2)

        elif pool == 'SPP':
            # NOTE: SPP does not reduce the resolution. It's output has 4 times the number of input channels.
            self.pool = SPP(channel)

        elif pool == 'None':
            self.pool = nn.Identity()

        else:
            raise NotImplementedError("Not expected pool: %s"%pool)
    
    def forward(self, x):
        return self.pool(x)
        
    
class Convolution(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, bias: bool = True, bn: bool = False, activation: str = 'ReLU'):
        super(Convolution, self).__init__()

        self.activation = Activation(activation)
        
        self.convolution = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch, affine=True, track_running_stats=True)
        else:
            self.bn = nn.Identity()
        
    def forward(self, x):
        return self.activation(self.bn(self.convolution(x)))
    
    
class Residual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size:int = 3, stride: int = 1, activation: str = 'ReLU'):
        super(Residual, self).__init__()
        
        self.activation = Activation(activation)
        
        self.conv1 = Convolution(in_ch, out_ch, kernel_size, stride, bias=False, bn=True, activation=activation)        
        self.conv2 = Convolution(out_ch, out_ch, kernel_size, stride, bias=False, bn=True, activation='Linear')
        
        if in_ch != out_ch:
            self.skip = Convolution(in_ch, out_ch, kernel_size=1, stride=stride, bias=False, bn=True, activation='Linear')
        else:
            self.skip = nn.Identity()
        
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return self.activation(y + self.skip(x))

    
class Hourglass(nn.Module):
    def __init__(self, num_layer: int, in_ch: int, increase_ch: int = 0, activation: str = 'ReLU', pool: str = 'Max'):
        super(Hourglass, self).__init__()
        mid_ch = in_ch + increase_ch 
        
        self.up1    = Residual(in_ch, in_ch, activation=activation)
        self.pool1  = Pool(in_ch, pool=pool)
        _in_ch = in_ch * 4 if pool == 'SPP' else in_ch

        self.low1 = Residual(_in_ch, mid_ch, activation=activation)
        # initialize the hourglass layers recursively
        if num_layer > 1:
            self.low2 = Hourglass(num_layer-1, mid_ch, increase_ch, activation=activation, pool=pool)
        else:
            self.low2 = Residual(mid_ch, mid_ch, activation=activation)

        self.low3 = Residual(mid_ch, in_ch, activation=activation)
        self.up2  =  nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        up1   = self.up1(x)
        pool1 = self.pool1(x)
        low1  = self.low1(pool1)
        low2  = self.low2(low1)
        low3  = self.low3(low2)
        up2   = self.up2(low3)
        return up1 + up2


class PreLayer(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 128, out_ch: int = 5, activation: str = 'ReLU', pool: str = 'Max'):
        super(PreLayer, self).__init__()
        layers = []
        layers.append(Convolution(in_ch=in_ch, out_ch=64, kernel_size=7, stride=2, bias=True, bn=True, activation=activation))
        layers.append(Residual(in_ch=64, out_ch=mid_ch))
        layers.append(Pool(channel=mid_ch, pool=pool))
        _mid_ch = mid_ch * 4 if pool == 'SPP' else mid_ch
        layers.append(Residual(in_ch=_mid_ch, out_ch=mid_ch))
        layers.append(Residual(in_ch=mid_ch, out_ch=out_ch))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Neck(nn.Module):
    def __init__(self, ch: int = 128, activation: str = 'ReLU', pool: str = 'None'):
        super(Neck, self).__init__()
        layers = []
        layers.append(Pool(ch, pool))
        layers.append(Convolution(in_ch=ch, out_ch=ch, kernel_size=1, bn=True, activation=activation))
        layers.append(Residual(ch, ch))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Head(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, stride: int = 1, bias: bool = True, bn: bool = False, activation: str = 'Linear'):
        super(Head, self).__init__()
        self.layer = Convolution(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride, bias=bias, bn=bn, activation=activation)

    def forward(self, x):
        return self.layer(x)

    
class StackedHourglass(nn.Module):
    def __init__(self, num_stack: int, in_ch: int, out_ch: int, increase_ch: int = 0, activation: str = 'ReLU', pool: str = 'Max', neck_activation: str = 'ReLU', neck_pool: str = 'None'):        
        super(StackedHourglass, self).__init__()

        # downsample the resolution of input (1 --> 1/4(scale_factor))
        self.pre_layer = PreLayer(in_ch=3, mid_ch=128, out_ch=in_ch, activation=activation, pool=pool)
        
        # hourglass modules (backbone)
        self.hourglass_lst = nn.ModuleList([Hourglass(num_layer=4, in_ch=in_ch, increase_ch=increase_ch, activation=activation, pool=pool) for _ in range(num_stack)])
        
        # feature layer (neck)
        self.neck_lst = nn.ModuleList([Neck(in_ch, neck_activation, neck_pool) for _ in range(num_stack)])
        
        # prediction layer (head)
        self.head_lst = nn.ModuleList([Head(in_ch=in_ch, out_ch=out_ch, kernel_size=1, stride=1, bias=True, bn=False, activation='Linear') for _ in range(num_stack)])
        
        # merge intermediate hourglass features
        self.merge_feature = nn.ModuleList([Convolution(in_ch=in_ch, out_ch=in_ch, kernel_size=1, stride=1, bias=True, bn=False, activation='Linear') for _ in range(num_stack-1)])
        
        # merger intermediate hourglass feature and prediction
        self.merge_prediction = nn.ModuleList([Convolution(in_ch=out_ch, out_ch=in_ch, kernel_size=1, stride=1, bias=True, bn=False, activation='Linear') for _ in range(num_stack-1)])

        self.num_stack = num_stack
        
    
    def forward(self, x):
        x = self.pre_layer(x)
        
        intermediate_predictions = []
        for i in range(len(self.hourglass_lst)):
            hg          = self.hourglass_lst[i](x)
            feature     = self.neck_lst[i](hg)            
            prediction  = self.head_lst[i](feature)

            intermediate_predictions.append(prediction)

            if i < len(self.hourglass_lst) - 1:
                x = x + self.merge_feature[i](feature) + self.merge_prediction[i](prediction)

        return torch.stack(intermediate_predictions, dim=1)


if __name__ == '__main__':
    # Stacked Hourglass module test
    stacked_hourglass = StackedHourglass(num_stack=2, in_ch=128, out_ch=5, increase_ch=0, activation='ReLU', pool='Max', neck_activation='ReLU', neck_pool='None')
    print(stacked_hourglass)
    stacked_hourglass.eval()
    x = torch.randn(2, 3, 512, 512)
    out = stacked_hourglass(x)
    num_param = sum([params.numel() for params in stacked_hourglass.parameters()])
    print('Stacked Hourglass (%d params) input: (%s), output: (%s)'%(num_param, x.shape, out.shape))

    # test jit
    scripted_sh = torch.jit.trace(stacked_hourglass, x)
    x2 = torch.ones(1, 3, 512, 512)
    out1 = stacked_hourglass(x2)
    x2 = torch.ones(1, 3, 512, 512)
    out2 = scripted_sh(x2)
    print('Jit test: ', torch.all(torch.eq(out1, out2)))
