# -*- coding: utf-8 -*-
"""
# @Date: 2020-05-15 11:30
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: demo.py
# Copyright @ 2020 yinhao. All rights reserved.
"""

import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import imageio
import os
from PIL import Image
import numpy as np
import math

# """将批处理的图片resize成固定的大小，用于后续处理"""
# def resize(img_dir, save_dir, scale):
#     for img_file in os.listdir(img_dir):
#         img_path = os.path.join(img_dir + os.sep, img_file)
#         if img_file == ".DS_Store":
#             os.remove(img_path)
#             continue
#         else:
#             img = Image.open(img_path)
#             w, h = img.size
#             new_image = img.resize((w*scale, h*scale), Image.BILINEAR)
#             new_image.save(os.path.join(save_dir, os.path.basename(img_file)))
#
#
# ### 新增代码用来测试单张图片的效果
# lr_dir = "/Users/yinhao/PycharmProjects/Meta-SR-Pytorch/benchmark/Set5/LR_bicubic/X2.00/"
# hr_dir = "/Users/yinhao/PycharmProjects/Meta-SR-Pytorch/benchmark/Set5/HR/"
# resize(lr_dir, hr_dir, 2)


def input_matrix_wpn_new(inH, inW, scale, add_scale=True):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''
    outH, outW = int(scale * inH), int(scale * inW)
    #### mask records which pixel is invalid, 1 valid or o invalid
    #### h_offset and w_offset caculate the offset to generate the input matrix
    scale_int = int(math.ceil(scale))
    h_offset = torch.ones(inH, scale_int, 1)
    mask_h = torch.zeros(inH, scale_int, 1)
    w_offset = torch.ones(1, inW, scale_int)
    mask_w = torch.zeros(1, inW, scale_int)


    ####projection  coordinate  and caculate the offset
    # h_project_coord = torch.arange(0, outH, 1).mul(1.0 / scale)
    ### 增加修改 torch.LongTensor->torch.FolatTensor
    h_project_coord = torch.arange(0, outH, 1).float().mul(1.0 / scale)
    # print(h_project_coord)
    # print(type(h_project_coord))
    ################################
    int_h_project_coord = torch.floor(h_project_coord)
    offset_h_coord = h_project_coord - int_h_project_coord
    int_h_project_coord = int_h_project_coord.int()

    # w_project_coord = torch.arange(0, outW, 1).mul(1.0 / scale)
    ### 增加修改 torch.LongTensor->torch.FolatTensor
    w_project_coord = torch.arange(0, outW, 1).float().mul(1.0 / scale)
    # print(w_project_coord)
    # print(type(w_project_coord))
    #################################
    int_w_project_coord = torch.floor(w_project_coord)
    offset_w_coord = w_project_coord - int_w_project_coord
    int_w_project_coord = int_w_project_coord.int()

    ####flag for   number for current coordinate LR image
    flag = 0
    number = 0
    for i in range(outH):
        if int_h_project_coord[i] == number:
            h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], flag, 0] = 1
            flag += 1
        else:
            h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], 0, 0] = 1
            number += 1
            flag = 1

    flag = 0
    number = 0
    for i in range(outW):
        if int_w_project_coord[i] == number:
            w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], flag] = 1
            flag += 1
        else:
            w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], 0] = 1
            number += 1
            flag = 1

    ## the size is scale_int* inH* (scal_int*inW)
    h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
    ####
    mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)


    mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(scale_int * inH, scale_int * inW)
    mask_mat = mask_mat.eq(2)

    i = 1
    h, w,_ = pos_mat.size()
    while(pos_mat[i][0][0]<= 1e-6 and i<h):
        i = i+1

    j = 1
    #pdb.set_trace()
    h, w,_ = pos_mat.size()
    while(pos_mat[0][j][1]<= 1e-6 and j<w):
        j = j+1

    pos_mat_small = pos_mat[0:i,0:j,:]

    pos_mat_small = pos_mat_small.contiguous().view(1, -1, 2)
    if add_scale:
        scale_mat = torch.zeros(1, 1)
        scale_mat[0, 0] = 1.0 / scale
        scale_mat = torch.cat([scale_mat] * (pos_mat_small.size(1)), 0)  ###(inH*inW*scale_int**2, 4)
        pos_mat_small = torch.cat((scale_mat.view(1, -1, 1), pos_mat_small), 2)

    return pos_mat_small, mask_mat  ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW



def prepare(args):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor

    return [_prepare(a) for a in args]



#########################
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)     ###setting the log and the train information

if checkpoint.ok:
    lr_img = "/Users/yinhao/PycharmProjects/Meta-SR-Pytorch/benchmark/Set5/LR_bicubic/X2.00/1.png"
    # device = torch.device(args.cpu)
    lr = imageio.imread(lr_img)   ### 读取图片
    lr = np.asarray(lr)           ### 将二进制转为numpy格式
    lr = torch.from_numpy(lr).half()    ### 将numpy数据转为tensor
    H, W, C = lr.size()
    # print(type(H))
    scale = 2
    # scale = args.scale
    outH, outW = int(H*scale), int(W*scale)
    scale_coord_map, mask = input_matrix_wpn_new(H, W, scale)

    model = model.Model(args, checkpoint)
    sr = model(lr, 0, scale_coord_map)
    # print(sr)
    re_sr = torch.masked_select(sr, mask)
    sr = re_sr.contiguous().view(C, outH, outW)
    sr = utility.quantize(sr, args.rgb_range)
    print(sr.shape)

    checkpoint.done()


