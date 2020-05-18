import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
# from dataset import TestDataset
from PIL import Image
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="carn")
    parser.add_argument("--ckpt_path", type=str, default="/Users/yinhao/PycharmProjects/CARN-pytorch/checkpoint/carn.pth")
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--sample_dir", type=str, default="./")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()


def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)

    im.save(filename)
    # im.save("/Users/yinhao_x/PycharmProjects/CARN-pytorch/save/2.jpg")


def sample(net, device, img_path, sr_im_path, cfg):
    t1 = time.time()
    # scale = int(cfg.scale)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    lr = Image.open(img_path).convert("RGB")
    lr = transform(lr)
    print(lr.shape)

    lr = lr.unsqueeze(0).to(device)

    sr = net(lr, cfg.scale).detach().squeeze(0)
    print(sr.shape)
    lr = lr.squeeze(0)
    t2 = time.time()
        
    # model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]

    save_image(sr, sr_im_path)
    print("Saved {} ({}x{} -> {}x{}, {:.3f}s)"
        .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1))


def main(cfg, img_path, sr_im_path):
    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.Net(multi_scale=True, 
                     group=cfg.group)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    state_dict = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict, )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    # print("net is:{}".format(net))

    sample(net, device, img_path, sr_im_path, cfg)
 

if __name__ == "__main__":
    cfg = parse_args()
    img_paths = "/Users/yinhao/PycharmProjects/CARN-pytorch/test/"
    save_path = "/Users/yinhao/PycharmProjects/CARN-pytorch/save/{}_x{}".format(cfg.model, cfg.scale)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for img_path in os.listdir(img_paths):
        img_name = img_path.split(".")[0]
        img_dir = os.path.join(img_paths+os.sep, img_path)
        sr_im_path = os.path.join(save_path+os.sep, "{}_SR.jpg".format(img_name))
        print(sr_im_path)
        main(cfg, img_dir, sr_im_path)
