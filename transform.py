import torch
import numpy as np

def box2hm(boxes, labels, imsize, scale_factor=4, num_cls=2, normalized=False):
    width, height   = imsize[0]//scale_factor, imsize[1]//scale_factor
    heat_map        = np.zeros((num_cls, height, width), dtype=np.float32)
    offset_map      = np.zeros((2, height, width), dtype=np.float32)
    size_map        = np.zeros((2, height, width), dtype=np.float32)
    mask            = np.zeros((1, height, width), dtype=np.float32)

    if boxes is None:
        return heat_map, offset_map, size_map, mask

    for box, label in zip(boxes, labels):
        if box is None:
            continue
        # sclae change (image to feature map)
        xmin, ymin, xmax, ymax = [val/scale_factor for val in box]
        
        # center point
        xcen, ycen = (xmax+xmin)/2, (ymax+ymin)/2
                
        # index of heat map
        xind, yind = int(xcen), int(ycen)        
        
        # set mask
        mask[:, yind, xind] = 1.0
        
        # offset, size
        xoff, yoff   = xcen - xind, ycen - yind
        xsize, ysize = xmax - xmin, ymax - ymin

        if normalized:
            xoff, yoff   = xoff/scale_factor, yoff/scale_factor
            xsize, ysize = xsize/width, ysize/height

        # assign offset, size and confidence
        offset_map[:, yind, xind] = np.array([xoff, yoff])
        size_map[:, yind, xind]   = np.array([xsize, ysize])
        
        # heatmap
        radius = ((xcen-xmin)**2 + (ycen-ymin)**2)**0.5
        draw_gaussian(heat_map[label], (xind, yind), radius)

    return heat_map, offset_map, size_map, mask


def gaussian2D(shape, sigma=1):
    m, n = list(map(int, shape))
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h


def draw_gaussian(heatmap, center, radius):
    gaussian = gaussian2D((radius, radius), sigma=radius/3)
    radius = int(radius)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def hm2box(heatmap, offset, wh, scale_factor=4, topk=10, conf_th=0.3, normalized=False):
    height, width = heatmap.shape[-2:]
    
    max_pool = torch.nn.MaxPool2d(3, stride=1, padding=3//2)        
    
    isPeak = max_pool(heatmap) == heatmap
    peakmap = heatmap * isPeak

    scores, indices = peakmap.flatten().topk(topk)
    
    clss  = torch.floor_divide(indices, (height*width))
    inds  = torch.fmod(indices, (height*width))
    yinds = torch.floor_divide(inds, width)
    xinds = torch.fmod(inds, width)
    
    xoffs = offset[0, yinds, xinds]
    xsizs = wh[0, yinds, xinds]
    
    yoffs = offset[1, yinds, xinds]
    ysizs = wh[1, yinds, xinds]
    
    if normalized:
        xoffs = xoffs * scale_factor
        yoffs = yoffs * scale_factor        
        xsizs = xsizs * width
        ysizs = ysizs * height
    
    xmin = (xinds + xoffs - xsizs/2) * scale_factor
    ymin = (yinds + yoffs - ysizs/2) * scale_factor
    xmax = (xinds + xoffs + xsizs/2) * scale_factor
    ymax = (yinds + yoffs + ysizs/2) * scale_factor
        
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1) # Tensor: topk x 4

    # confidence thresholding
    over_threshold = scores >= conf_th

    return boxes[over_threshold], clss[over_threshold], scores[over_threshold]

if __name__ == '__main__':
    boxes = [[10, 20, 100, 200]]
    labels = [1]
    imsize = 512, 512
    normalized = True
    print('original box: ', boxes)

    # numpy
    heatmap_np, offset_np, wh_np, mask = box2hm(boxes, labels, imsize, normalized=normalized)
    print('Shape of heatmap: ', heatmap_np.shape)
    print('Value of heatmap: ', heatmap_np[:, 27, 13])
    print('Value of offset: ', offset_np[:, 27, 13])
    print('Value of wh: ', wh_np[:, 27, 13])

    # numpy to tensor
    heatmap_ten = torch.from_numpy(heatmap_np)
    offset_ten  = torch.from_numpy(offset_np)
    wh_ten      = torch.from_numpy(wh_np)
    _boxes, _labels, _scores = hm2box(heatmap_ten, offset_ten, wh_ten, normalized=normalized)
    print('re-calculated box: ', _boxes.tolist())
