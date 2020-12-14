import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pdb
import math

class Grid(object):
    def __init__(self, d1, d2, device, ratio = 0.5, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.ratio = ratio
        self.device = device
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, x):
        if np.random.rand() > self.prob:
            return x
        h = x.shape[2]
        w = x.shape[3]
        z = x.shape[1]
        
        st_s = np.random.randint(z)
        step_s = np.random.randint(1,3)
        
        mask_cube = torch.ones(z,h,w).to(self.device)
        
        d = np.random.randint(self.d1, self.d2)
        self.l = math.ceil(d*self.ratio)
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image 
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h*h + w*w)))
        
        #for j in range(st_s, z, step_s):
        for j in range(z):

            mask = np.ones((hh, hh), np.float32)
            st_h = np.random.randint(d)
            st_w = np.random.randint(d)

            for i in range(-1, hh//d+1):
                    s = d*i + st_h
                    t = s+self.l
                    s = max(min(s, hh), 0)
                    t = max(min(t, hh), 0)
                    mask[s:t,:] *= 0
            for i in range(-1, hh//d+1):
                    s = d*i + st_w
                    t = s+self.l
                    s = max(min(s, hh), 0)
                    t = max(min(t, hh), 0)
                    mask[:,s:t] *= 0
                    
            mask = np.asarray(mask)
            mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]
            
            mask = torch.from_numpy(mask).float().to(self.device)
            mask = 1-mask
            mask = mask.expand(h,w)
            mask_cube[j] = mask
        return x*mask_cube
    
class GridMask(nn.Module):
    def __init__(self, d1, d2, device, ratio = 0.5, prob=1., max_epoch = 10):
        super(GridMask, self).__init__()
        self.ratio = ratio
        self.st_prob = prob
        self.max_epoch = max_epoch
        self.grid = Grid(d1, d2, device, ratio, prob)

    def set_prob(self, epoch):
        self.grid.set_prob(epoch, self.max_epoch)

    def get_prob(self):
        return self.grid.prob

    def forward(self, x):
        if not self.training:
            return x
        n,c,s,h,w = x.size()
        y = []
        y_s = []
        for i in range(n):
            y.append(self.grid(x[i]))
        y = torch.cat(y).view(n,c,s,h,w)
        return y