import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict
from cuda.shift import Shift

import slidingwindow as sw


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class shift_conv_shift(nn.Module):
    def __init__(self, channel, stride=1, init_scale=1):
        super(shift_conv_shift, self).__init__()
        self.bn_1 = nn.BatchNorm2d(channel).cuda()
        self.shift_in = Shift(channel=channel, stride=1, init_scale=1)
        self.linear = nn.Conv2d(channel, channel, 1).cuda()
        nn.init.kaiming_normal(self.linear.weight, mode='fan_out')
        self.relu = nn.ReLU(inplace=True)
        self.shift_out = Shift(channel=channel, stride=1, init_scale=1)
        self.bn_2 = nn.BatchNorm2d(channel).cuda()
        bn_init(self.bn_2, 1)
    def forward(self, x):
        x = self.bn_2(self.shift_out(self.relu(self.linear(self.shift_in(self.bn_1(x))))))
        return x
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)   #AdaptiveAvgPool2d
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return y.expand_as(x)

def  bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)

class Shift_stcn_block(nn.Module):
    def __init__(self, in_channels, out_channels, group_channels_s, group_channels_t,kernel_size=9, stride=1):
        super(Shift_stcn_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_channels_s = group_channels_s
        self.group_channels_t = group_channels_t
###################################  cross atts  ##############################################
        self.se_s1 = SELayer(channel=group_channels_t, reduction=16)
        self.se_s2 = SELayer(channel=group_channels_t, reduction=16)    #  concated h-shift and w-hift features
        self.se_t1 = SELayer(channel=group_channels_t, reduction=16)
        self.se_t2 = SELayer(channel=group_channels_t, reduction=16)
        self.se_s1.cuda()
        self.se_s2.cuda()
        self.se_t1.cuda()
        self.se_t2.cuda()
##########################################################################################
        self.scs_h_s1_p1 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1) # scale1_patch_1
        self.scs_h_s1_p2 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
        self.scs_h_s1_p3 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
        self.scs_h_s1_p4 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
        self.scs_h_s2_p1 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
##############################################################################################
        self.scs_w_s1_p1 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1) # scale1_patch_1
        self.scs_w_s1_p2 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
        self.scs_w_s1_p3 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
        self.scs_w_s1_p4 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
        self.scs_w_s2_p1 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
##################################################################################
        self.scs_t_s1_p1 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
        self.scs_t_s2_p1 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
        self.scs_t_s2_p2 = shift_conv_shift(channel=group_channels_s, stride=1, init_scale=1)
##########################################################################################################        

    def forward(self, x):
        B, C, T, H, W = x.size()
        group_size = int(np.floor(C/4))
        x_t = x[:,group_size*2:group_size*3,:,:,:]
        x_unchange = x[:,group_size*3:,:,:,:]       
        y = x.permute(1, 3, 4, 0, 2).contiguous().view(C, H, W, B*T)
        y = y.cpu().detach().numpy()  
        yy  = np.zeros((C, H+1, W+1, B*T))
        yy[:,:-1,:-1,:] = y
#######################################   s_scale_1_start   #############################################################         
        S = int(np.floor((H+1)/2))
        windows = sw.generate(yy, sw.DimOrder.ChannelHeightWidth, S, 0)
        block_list=[]
        for i, window in enumerate(windows):
            temp = window.apply(yy)
            temp = torch.from_numpy(temp).cuda()
            temp = temp.type(torch.cuda.FloatTensor)
            temp = temp.view(C, S, S, B, T).permute(3,0,4,1,2).contiguous()   # B, C, T, H, W (S)
            # channel split
            temp0 = temp[:,:group_size,:,:,:]
            temp1 = temp[:,group_size:group_size*2,:,:,:]
            # scale_1
            xh = temp0.permute(0,1,3,2,4).contiguous().view(B, group_size, S, T* S )
            xw = temp1.permute(0,1,4,2,3).contiguous().view(B, group_size, S, T* S )
            if i == 0:
                xh = self.scs_h_s1_p1(xh)
                xw = self.scs_w_s1_p1(xw)
            elif i ==1:
                xh = self.scs_h_s1_p2(xh)
                xw = self.scs_w_s1_p2(xw)
            elif i ==2:
                xh = self.scs_h_s1_p3(xh)
                xw = self.scs_w_s1_p3(xw)
            elif i ==3:
                xh = self.scs_h_s1_p4(xh)
                xw = self.scs_w_s1_p4(xw)
            xh= xh.view(B, group_size, S, T, S).permute(0,1,3,2,4).contiguous()
            xw = xw.view(B, group_size, S, T, S).permute(0,1,3,4,2).contiguous()     

            tempblock = torch.cat([xh,xw], dim=1)       # concat
            block_list.append(tempblock)   

        block_0 = block_list[0]
        block_1 = block_list[1]
        block_2 = block_list[2]
        block_3 = block_list[3]


        # 4 blocks
        block_a =  torch.cat([block_0,block_1], dim=3)
        block_b =  torch.cat([block_2,block_3], dim=3)

        block_scale1 = torch.cat([block_a,block_b], dim=4)
#######################################   s_scale_1_end &s_scale_2_start  ##################

        yy = torch.from_numpy(yy).cuda()
        yy = yy.type(torch.cuda.FloatTensor)
        yy = yy.view(C, H+1, W+1, B, T).permute(3,0,4,1,2).contiguous()
        yy_0 = yy[:,:group_size,:,:,:]
        yy_1 = yy[:,group_size:group_size*2,:,:,:]
        yy_h = yy_0.permute(0,1,3,2,4).contiguous().view(B, group_size, (H+1), T*(W+1 ))
        yy_h = self.scs_h_s2_p1(yy_h)
        yy_h= yy_h.view(B, group_size, (H+1), T, W+1).permute(0,1,3,2,4).contiguous()
        yy_w = yy_1.permute(0,1,4,2,3).contiguous().view(B, group_size, W+1, T*(H+1) )
        yy_w = self.scs_w_s2_p1(yy_w)
        yy_w = yy_w.view(B, group_size, W+1, T, (H+1)).permute(0,1,3,4,2).contiguous()     
        
        block_scale2 = torch.cat([yy_h,yy_w], dim=1)

#######################################   s_scale_2_end & s_cross_attention_start  #################################
        s1_att = self.se_s1(block_scale1)   
        s2_att = self.se_s2(block_scale2)
        block_scale1 = block_scale1 * s2_att 
        block_scale2 = block_scale2 * s1_att 
#######################################   s_cross_attention_end   #############################################################         
        blocks = block_scale1 + block_scale2  
        blocks = blocks / 2    # mean
        blocks=blocks[:,:,:,:-1,:-1]
  
        xt = x_t.view(B, group_size, T, H*W)
        temporal_size = int(np.floor(T/2))
        xt_1 = xt[:, :, :temporal_size,:]
        xt_2 = xt[:, :, temporal_size:, :]
        
        #######################################   t_scale_1_start   #############################################################        
        xt_1 = self.scs_t_s2_p1(xt_1)
        xt_2 = self.scs_t_s2_p2(xt_2)

        xt12 = torch.cat([xt_1,xt_2], dim=2)
        xt12 = xt12.view(B, group_size, T, H, W)
        xt12 = torch.cat([xt12, x_unchange], dim=1)

#######################################   t_scale_1_end &  t_scale_2_start #############################################################
        xt = self.scs_t_s1_p1(xt)
        xt = xt.view(B, group_size, T, H, W)
        xt = torch.cat([xt, x_unchange], dim=1)

#######################################   t_scale_2_end & t_cross_attention_start   #############################################################
        t1_att = self.se_t1(xt12)
        t2_att = self.se_t2(xt)
        xt12 = xt12 * t2_att                 #+ xt12
        xt = xt * t1_att                      # + xt
#######################################   t_cross_attention_end   ####################################
        xts = (xt12 + xt)/2
        blocks = blocks.cuda()

        xts= xts.cuda()
        xx = torch.cat([blocks,xts], dim=1)  
        x = x.cuda()
        xx = xx.cuda()
        x = x+xx # residual connection
        return x


class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        self.conv3d.cuda()
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)
            self.bn.cuda()
    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        
        x = x.to(device)
        x = self.conv3d(x)
        x  =  x.to(device)
        if self._use_batch_norm:
            x = x.to(device)
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
       'Temp_shift_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Temp_shift_5c'
        self.end_points[end_point] = Shift_stcn_block(384+384+128+128, 384+384+128+128, 256, 512, stride=1)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits  = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x, pretrained=False, n_tune_layers=-1):
        if pretrained:
            assert n_tune_layers >= 0

            freeze_endpoints = self.VALID_ENDPOINTS[:-n_tune_layers]
            tune_endpoints = self.VALID_ENDPOINTS[-n_tune_layers:]
        else:
            freeze_endpoints = []
            tune_endpoints = self.VALID_ENDPOINTS

        # backbone, no gradient part
        with torch.no_grad():
            for end_point in freeze_endpoints:
                if end_point in self.end_points:
                    x = self._modules[end_point](x) # use _modules to work with dataparallel

        # backbone, gradient part
        for end_point in tune_endpoints:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        # head
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits
        

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)