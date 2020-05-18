import os
import json
import time
import importlib
import argparse
from collections import OrderedDict
import torch
from dataset import TestDataset
from PIL import Image
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,default="mcan-fast")
    parser.add_argument("--ckpt_path", type=str, default="/Users/yinhao/PycharmProjects/MCAN/checkpoint/mcan_91000.pth")
    parser.add_argument("--sample_dir", type=str, default="sample/")
    parser.add_argument("--sample_scale", type=int, default=2)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--shave", type=int, default=20)
    # parser.add_argument("--sample_data_set", type=str, default="calculate_sets_xx/Set5")
    return parser.parse_args()


def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)

    im.save(filename)


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
    sr = net(lr, cfg.sample_scale).detach().squeeze(0)
    print(sr.shape)
    lr = lr.squeeze(0)
    t2 = time.time()

    # lr = lr.unsqueeze(0).to(device)
    # sr = net(lr, cfg.sample_scale).detach().squeeze(0)
    # print(sr.shape)
    # lr = lr.squeeze(0)
    # t2 = time.time()

    # model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]

    save_image(sr, sr_im_path)
    print("Saved {} ({}x{} -> {}x{}, {:.3f}s)"
          .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1))



def main(cfg, img_path, sr_im_path):
    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.Net(multi_scale=True,
                     group=cfg.group)
    # net = module.Net(scale=cfg.sample_scale, group=cfg.group)

    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    # if cfg.sample_scale > 0:
    #    net = model(scale=cfg.sample_scale, group=cfg.group)
    # else:
    #    net = model(multi_scale=True, group=cfg.group)

    state_dict = torch.load(cfg.ckpt_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # if cfg.sample_scale > 0:
    sample(net, device, img_path, sr_im_path, cfg)
    # else:
    #     for s in range(2, 5):
    #         cfg.sample_scale = s
    #         sample(net, device, img_path, sr_im_path, cfg)


if __name__ == "__main__":
    cfg = parse_args()
    img_paths = "/Users/yinhao/PycharmProjects/MCAN/test/"
    save_path = "/Users/yinhao/PycharmProjects/MCAN/save/{}_x{}_91000".format(cfg.model, cfg.sample_scale)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for img_path in os.listdir(img_paths):
        img_name = img_path.split(".")[0]
        img_dir = os.path.join(img_paths+os.sep, img_path)
        sr_im_path = os.path.join(save_path+os.sep, "{}_SR.jpg".format(img_name))
        print(sr_im_path)
        main(cfg, img_dir, sr_im_path)
