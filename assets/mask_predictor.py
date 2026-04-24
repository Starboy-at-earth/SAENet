import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import sys
from einops import rearrange, repeat
from .cst import cst
from .text_encoder.text_encoder import TextEncoder, FeatureResizer

def generate_coord(batch, height, width):

    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    #print(batch, height, width)
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord


class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims=256, factor=2):
        super(SimpleDecoding, self).__init__()
        hidden_size = 256
        c4_size = 256
        c3_size = 256
        c2_size = 256
        c1_size = 256
        c0_size = 256

        self.conv1_3 = nn.Conv2d(hidden_size*3, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size*3, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size*3, hidden_size, 3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(hidden_size)
        self.relu1_1 = nn.ReLU()

        self.conv2_1 = nn.Sequential(nn.Conv2d(hidden_size*2, hidden_size, 3, padding=1, bias=False),
                         nn.BatchNorm2d(hidden_size),
                         nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(hidden_size*2, hidden_size, 3, padding=1, bias=False),
                         nn.BatchNorm2d(hidden_size),
                         nn.ReLU())
        


    def forward(self, x_c3, x_c2, x_c1, x_c0):
        

        x_c2_c0 = F.interpolate(input=x_c2, size=(x_c0.size(-2), x_c0.size(-1)), mode='bilinear', align_corners=True)
        x_c2_c0 = abs(x_c0 - x_c2_c0)
        x_c1_c0 = F.interpolate(input=x_c1, size=(x_c0.size(-2), x_c0.size(-1)), mode='bilinear', align_corners=True)
        x_c1_c0 = abs(x_c0 - x_c1_c0)
        x_c0_tmp = torch.cat([x_c0, x_c2_c0, x_c1_c0], dim=1)
        x_c0_tmp = self.conv1_3(x_c0_tmp)
        x_c0_tmp = self.bn1_3(x_c0_tmp)
        x_c0_tmp = self.relu1_3(x_c0_tmp)

        x_c2_c1 = F.interpolate(input=x_c2, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x_c2_c1 = abs(x_c1 - x_c2_c1)
        x_c0_c1 = F.interpolate(input=x_c0, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x_c0_c1 = abs(x_c1 - x_c0_c1)
        x_c1_tmp = torch.cat([x_c1, x_c2_c1, x_c0_c1], dim=1)
        x_c1_tmp = self.conv1_2(x_c1_tmp)
        x_c1_tmp = self.bn1_2(x_c1_tmp)
        x_c1_tmp = self.relu1_2(x_c1_tmp)

        x_c0_c2 = F.interpolate(input=x_c0, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x_c0_c2 = abs(x_c2 - x_c0_c2)
        x_c1_c2 = F.interpolate(input=x_c1, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x_c1_c2 = abs(x_c2 - x_c1_c2)
        x_c2_tmp = torch.cat([x_c2, x_c1_c2, x_c0_c2], dim=1)
        x_c2_tmp = self.conv1_1(x_c2_tmp)
        x_c2_tmp = self.bn1_1(x_c2_tmp)
        x_c2_tmp = self.relu1_1(x_c2_tmp)


        x_c2_tmp = F.interpolate(input=x_c2_tmp, size=(x_c1_tmp.size(-2), x_c1_tmp.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c2_tmp, x_c1_tmp], dim=1)
        x = self.conv2_1(x)

        x = F.interpolate(input=x, size=(x_c0_tmp.size(-2), x_c0_tmp.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c0_tmp, x], dim=1)
        x = self.conv2_2(x)
        
        return x

class FuseDecoding(nn.Module):
    def __init__(self, c4_dims=256, factor=2):
        super(FuseDecoding, self).__init__()
        self.h2l = nn.MultiheadAttention(c4_dims, 8, dropout=0.1)
        hidden_size = c4_dims
        self.cst = cst(256, 256, 0.1)
        pass
        
    def forward(self, low, high, lang_feat, lang_mask):

        shortcut = low
        b, t ,c, hl, wl = low.size()
        b, t ,c, hh, wh = high.size()
        

        low = rearrange(low, 'b t c h w -> (h w) (b t) c')
        high = rearrange(high, 'b t c h w -> (h w) (b t) c')

        response = self.h2l(query=low,
                            key=high,
                            value=high, attn_mask=None,
                            key_padding_mask=None)[0]
        low = rearrange(low, '(h w) (b t) c -> (b t) c h w', h = hl, w = wl, b=b, t=t)
        response = rearrange(response, '(h w) (b t) c -> (b t) c h w', h = hl, w = wl, b=b, t=t)

        x = self.cst(low, response)
        x = rearrange(x, '(b t) c h w -> b t c h w', h = hl, w = wl, b=b, t=t)

        return x 


