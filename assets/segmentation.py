"""
Segmentaion Part 
Modified from DETR (https://github.com/facebookresearch/detr)
"""
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange, repeat

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

import fvcore.nn.weight_init as weight_init

from .position_encoding import PositionEmbeddingSine1D
from .text_encoder.text_encoder import TextEncoder, FeatureResizer

BN_MOMENTUM = 0.1

def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    #print(batch, height, width)
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

def get_norm(norm, out_channels): # only support GN or LN
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(8, channels),
            "LN": lambda channels: nn.LayerNorm(channels)
        }[norm]
    return norm(out_channels)

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)

class SIM(nn.Module):
    def __init__(self, h_C, l_C):
        super(SIM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = cus_sample

        self.h2l_0 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.h2h_0 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.bnl_0 = nn.BatchNorm2d(l_C)
        self.bnh_0 = nn.BatchNorm2d(h_C)

        self.h2h_1 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(l_C)
        self.bnh_1 = nn.BatchNorm2d(h_C)

        self.h2h_2 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.l2h_2 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.bnh_2 = nn.BatchNorm2d(h_C)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        h, w = x.shape[2:]
        # first conv
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))
        x_l = self.relu(self.bnl_0(self.h2l_0(self.h2l_pool(x))))

        # mid conv
        x_h2h = self.h2h_1(x_h)
        x_h2l = self.h2l_1(self.h2l_pool(x_h))
        x_l2l = self.l2l_1(x_l)
        x_l2h = self.l2h_1(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))

        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_2(x_h2h + x_l2h))

        return x_h + x

class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        
        self.fuse_l2v_0 =FeatureResizer(
            input_feat_size=d_model*2+8,
            output_feat_size=d_model,
            dropout=0.1,
        )
#######################################################################
        self.multihead_attn_image10 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_image11 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.row = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.fuse_row = FeatureResizer(
            input_feat_size=d_model*2,
            output_feat_size=d_model,
            dropout=0.1,
        )
#######################################################################
        self.multihead_attn_image20 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_image21 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.col = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.fuse_col = FeatureResizer(
            input_feat_size=d_model*2,
            output_feat_size=d_model,
            dropout=0.1,
        )
        
        self.init_weights()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    nn.init.xavier_uniform_(m.weight, gain=1)
                    
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight,1)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=('upernet' in pretrained), logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, visual, text,
                text_key_padding_mask: Optional[Tensor] = None,
                text_pos: Optional[Tensor] = None,
                visual_pos: Optional[Tensor] = None):
        ############################## Language2Image_stage0 ##############################
        
        t, h, w, b, c = visual.size()
        coord = generate_coord(b*t, h, w) # b*t, 8, h, w
        coord = rearrange(coord, '(b t) c h w -> (t h w) b c', t=t)
        visual = self.with_pos_embed(visual, visual_pos)
        visual = rearrange(visual, 't h w b c -> (t h w) b c')
        visual = self.fuse_l2v_0(torch.cat([visual * text[0,:,:].unsqueeze(dim=0), visual, coord], dim=2))
    
        ############################## Language2Image_stage1 ##############################
        visual_1 = rearrange(visual, '(t h w) b c -> t (h w) b c', t=t, h=h, w=w)
        visual_vector_1 = (torch.mean(visual_1, dim=1))
        lang_feature_1 = self.multihead_attn_image10(query=visual_vector_1,
                                   key=self.with_pos_embed(text, text_pos),
                                   value=text, attn_mask=None,
                                   key_padding_mask=text_key_padding_mask)[0]
        lang_feature_1 = self.multihead_attn_image11(query=lang_feature_1,
                                   key=lang_feature_1,
                                   value=lang_feature_1, attn_mask=None,
                                   key_padding_mask=None)[0]
        visual_attended_1 = visual_1*lang_feature_1.unsqueeze(dim=1)
        visual_1 = rearrange(visual_1, 't (h w) b c -> (t h w) b c', t=t, h=h, w=w)
        visual_attended_1 = rearrange(visual_attended_1, 't (h w) b c -> (t h w) b c', t=t, h=h, w=w)
        
        #########################################
        visual_image_row_query = rearrange(visual_1, '(t h w) b c-> w (t h b) c', t=t, h=h, w=w, b =b ,c=c)
        visual_image_row_key_value =  rearrange(visual_attended_1, '(t h w) b c-> w (t h b) c', t=t, h=h, w=w, b =b ,c=c)
        visual_image_row_response = self.row(query=visual_image_row_query,
                                key=visual_image_row_key_value,
                                value=visual_image_row_key_value, attn_mask=None,
                                key_padding_mask=None)[0]
        visual_1 = self.fuse_row(torch.cat([visual_image_row_query, visual_image_row_response], dim=2))

        ##############################Image Level##############################
        visual_2 = rearrange(visual_1, 'w (t h b) c -> t (h w) b c', t=t, h=h, w=w)
        visual_vector_2 = (torch.mean(visual_2, dim=1))
        lang_feature_2 = self.multihead_attn_image20(query=visual_vector_2,
                                   key=self.with_pos_embed(text, text_pos),
                                   value=text, attn_mask=None,
                                   key_padding_mask=text_key_padding_mask)[0]
        lang_feature_2 = self.multihead_attn_image21(query=lang_feature_2,
                                   key=lang_feature_2,
                                   value=lang_feature_2, attn_mask=None,
                                   key_padding_mask=None)[0]
        visual_attended_2 = visual_2*lang_feature_2.unsqueeze(dim=1)
        visual_2 = rearrange(visual_2, 't (h w) b c -> (t h w) b c', t=t, h=h, w=w)
        visual_attended_2 = rearrange(visual_attended_2, 't (h w) b c -> (t h w) b c', t=t, h=h, w=w)
        
        ###########################################
        visual_image_col_query = rearrange(visual_2, '(t h w) b c -> h (t w b) c', t=t, h=h, w=w, b =b ,c=c)
        visual_image_col_key_value = rearrange(visual_attended_2, '(t h w) b c -> h (t w b) c', t=t, h=h, w=w, b =b ,c=c)
        visual_image_col_response = self.col(query=visual_image_col_query,
                                key=visual_image_col_key_value,
                                value=visual_image_col_key_value, attn_mask=None,
                                key_padding_mask=None)[0]
        visual_2 =  self.fuse_col(torch.cat([visual_image_col_query, visual_image_col_response], dim=2))
        visual_2 =  rearrange(visual_2, 'h (t w b) c -> (t h w) b c', t=t, h=h, w=w, b =b ,c=c)
        return visual_2


class VisionLanguageFusionModule2(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_image1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.fuse0 =FeatureResizer(
            input_feat_size=d_model*2+8,
            output_feat_size=d_model,
            dropout=0.1,
        )
        self.fuse1 = FeatureResizer(
            input_feat_size=d_model*2,
            output_feat_size=d_model,
            dropout=0.1,
        )
       
        self.init_weights()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    nn.init.xavier_uniform_(m.weight, gain=1)
                    
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight,1)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=('upernet' in pretrained), logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, visual, text,
                text_key_padding_mask: Optional[Tensor] = None,
                text_pos: Optional[Tensor] = None,
                visual_pos: Optional[Tensor] = None):
        t, h, w, b, c = visual.size()
        coord = generate_coord(b*t, h, w) # b*t, 8, h, w
        coord = rearrange(coord,'(b t) c h w -> (t h w) b c', t=t)

        visual = rearrange(visual, 't h w b c -> (t h w) b c')
        visual = self.fuse0(torch.cat([visual * text[0,:,:].unsqueeze(dim=0), visual, coord], dim=2))
        
        ##############################Image Level##############################
        visual_image_level = rearrange(visual, '(t h w) b c -> t (h w) b c', t=t, h=h, w=w)
        visual_image_vec = (torch.mean(visual_image_level, dim=1))
        visual_image_feature = self.multihead_attn_image(query=visual_image_vec,
                                   key=self.with_pos_embed(text, text_pos),
                                   value=text, attn_mask=None,
                                   key_padding_mask=text_key_padding_mask)[0]
        visual_image_feature = self.multihead_attn_image1(query=visual_image_feature,
                                   key=visual_image_feature,
                                   value=visual_image_feature, attn_mask=None,
                                   key_padding_mask=None)[0]
        visual_image_level_attended = visual_image_level*visual_image_feature.unsqueeze(dim=1)
        visual_image_level = rearrange(visual_image_level, 't (h w) b c -> (t h w) b c', t=t, h=h, w=w)
        visual_image_level_attended = rearrange(visual_image_level_attended, 't (h w) b c -> (t h w) b c', t=t, h=h, w=w)
        visual_image_level = self.fuse1(torch.cat([visual_image_level_attended, visual_image_level], dim=2))
        return visual_image_level

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

