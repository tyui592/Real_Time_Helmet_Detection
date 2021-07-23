import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision
import torchvision.transforms.functional as TF

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.sum    += val * n
        self.count  += n
        self.avg    = self.sum / self.count

def ten2pil(tensor, pretrained):
    if pretrained is None:
        denormalize = lambda x: x
    else:
        denormalize = get_normalizer(denormalize = True, pretrained = pretrained)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor, pad_value=0.5)
    image  = TF.to_pil_image(denormalize(tensor).clamp_(0.0, 1.0))
    return image

def draw_box(pil, box, width=2, color=(0, 0, 255)):
    draw = ImageDraw.Draw(pil)
    draw.rectangle(list(map(int, box)), width=width, outline=color, fill=None)
    return pil

def write_text(pil, text, coordinate, fontsize=15, fontcolor='red'):
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype('arial.ttf', size=fontsize)
    draw.text(coordinate, text, fill=fontcolor, font=font)
    return pil

def get_normalizer(pretrained, denormalize = False):
    if pretrained.lower() == "imagenet":
        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]
    elif pretrained.lower() == "scratch":
        MEAN = [0.5, 0.5, 0.5]
        STD  = [0.5, 0.5, 0.5]
    else:
        raise NotImplementedError("Not expected dataset pretrained parameter: %s"%pretrained)

    if denormalize:
        MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
        STD  = [1/std for std in STD]
    return torchvision.transforms.Normalize(mean=MEAN, std=STD)

def blend_heatmap(image, heatmap, pretrained):
    image_pil = ten2pil(image.detach().cpu(), pretrained=pretrained)

    for c in range(heatmap.shape[0]):
        heatmap_rgb = [np.zeros(heatmap.shape[1:], dtype=np.uint8)]*2

        _heatmap = heatmap[c]
        _heatmap_np = _heatmap.detach().cpu().numpy() * 255
        _heatmap_np = _heatmap_np.astype(np.uint8)

        # gray to rgb
        heatmap_rgb.insert(c, _heatmap_np)

        heatmap_pil = Image.fromarray(np.stack(heatmap_rgb, axis=-1)).resize(image_pil.size).convert('RGB')
        image_pil = Image.blend(image_pil, heatmap_pil, 0.3)
    return image_pil

def imload(path, pretrained, size=None):
    img_pil = Image.open(path).convert('RGB')
    origin_size = img_pil.size
    if size:
        img_pil = img_pil.resize((size, size))
    normalizer = get_normalizer(pretrained=pretrained)
    img_ten = normalizer(TF.to_tensor(img_pil)).unsqueeze(0)
    return img_ten, img_pil, origin_size
