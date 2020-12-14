import torch
import torch.nn as nn


## Code adapted from : https://github.com/digantamisra98/EvoNorm ##
## Data expected to be in format B,C,D,W,H ##
def instance_std_3D(x, eps=1e-5):
    var = torch.var(x, dim = (2, 3, 4), keepdim=True).expand_as(x)
    return torch.sqrt(var + eps)


def group_std_3D(x, groups = 8, eps = 1e-5, version='S0'):
    N, C, D, W, H = x.size()
    if (version == 'S0'):
        if (C // groups == 0):
            var = torch.var(x, dim = (1, 3, 4), keepdim = True).expand_as(x)
        else:
            x = torch.reshape(x, (N, groups, C // groups, D, W, H))
            var = torch.var(x, dim = (2, 4, 5), keepdim = True).expand_as(x)
    elif (version == 'S0_3D'):
        if (C // groups == 0):
            var = torch.var(x, dim = (1, 2, 3, 4), keepdim = True).expand_as(x)
        else:
            x = torch.reshape(x, (N, groups, C // groups, D, W, H))
            var = torch.var(x, dim = (2, 3, 4, 5), keepdim = True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, D, W, H))

def instance_std_2D(x, eps=1e-5):
    var = torch.var(x, dim = (2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)

def group_std_2D(x, groups = 8, eps = 1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))


class EvoNorm2D(nn.Module):

    def __init__(self, input, non_linear = True, version = 'S0', affine = True, momentum = 0.9, eps = 1e-5):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.momentum = momentum
        self.eps = eps
        if self.version not in ['B0', 'S0']:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
            if self.non_linear:
                self.v = nn.Parameter(torch.ones(1,self.insize,1,1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            self.register_buffer('v', None)
        self.register_buffer('running_var', torch.ones(1, self.insize, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
    
    def forward(self, x):
        self._check_input_dim(x)
        if self.version == 'S0':
            if self.non_linear:
                num = x * torch.sigmoid(self.v * x)
                return num / group_std_2D(x, eps = self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                var = torch.var(x, dim = (0, 2, 3), unbiased = False, keepdim = True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max((var+self.eps).sqrt(), self.v * x + instance_std_2D(x, eps = self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta

class EvoNorm3D(nn.Module):

    def __init__(self, input, sequence = 5, non_linear = True, version = 'S0', momentum = 0.9):
        super(EvoNorm3D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.eps = 1e-5
        self.momentum = momentum
        if self.version not in ['B0', 'S0', 'B0_3D', 'S0_3D']:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        if self.version in ['B0', 'S0']: 
            self.gamma = nn.Parameter(torch.ones(1, self.insize, sequence, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.insize, sequence, 1, 1))
            if self.non_linear:
                self.v = nn.Parameter(torch.ones(1,self.insize, sequence, 1,1))
            self.register_buffer('running_var', torch.ones(1, self.insize, sequence, 1, 1))
        elif self.version in ['B0_3D', 'S0_3D']:
            self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1, 1))
            if self.non_linear:
                self.v = nn.Parameter(torch.ones(1,self.insize, 1, 1, 1))
            self.register_buffer('running_var', torch.ones(1, self.insize, 1, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)
    
    def forward(self, x):
        if x.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(x.dim()))
        if self.version == 'S0' or self.version == 'S0_3D':
            if self.non_linear:
                num = x * torch.sigmoid(self.v * x)
                if (self.version == 'S0'):
                    return num / group_std_3D(x) * self.gamma + self.beta
                elif (self.version == 'S0_3D'):
                    return num / group_std_3D(x, version='S0_3D') * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0' or self.version == 'B0_3D':
            if self.training:
                if (self.version == 'B0'):
                    var = torch.var(x, dim = (0, 3, 4), unbiased = False, keepdim = True).reshape(1, x.size(1), x.size(2), 1, 1)
                else:
                    var = torch.var(x, dim = (0, 2, 3, 4), unbiased = False, keepdim = True).reshape(1, x.size(1), 1, 1, 1)
                with torch.no_grad():
                    self.running_var.copy_(self.momentum * self.running_var + (1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max((var+self.eps).sqrt(), self.v * x + instance_std_3D(x))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta