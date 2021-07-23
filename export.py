import torch
import torch.nn as nn
import torchvision

from transform import hm2box
from utils import get_normalizer

class Export(torch.nn.Module):
    def __init__(self, network, topk, scale_factor, conf_th, nms_th, normalized_coord=False):
        super(Export, self).__init__()

        self.network          = network
        self.topk             = topk
        self.scale_factor     = scale_factor
        self.conf_th          = conf_th
        self.nms_th           = nms_th
        self.normalized_coord = normalized_coord

    def forward(self, x):
        # x: input tensor (1, c, h, w)
        # output: 1, n, 6, h, w
        batch_output = self.network(x)

        # batch-wise for loop
        for outputs in batch_output.split(1, dim=0):

            # stack-wise for loop
            stack_boxes, stack_clss, stack_scores = [], [], []
            for output in outputs.split(1, dim=1):
                output.squeeze_(1)
                heatmap, offset, wh = output.split([2,2,2], dim=1)
                heatmap = torch.sigmoid(heatmap)
                if self.normalized_coord:
                    offset = torch.sigmoid(offset)
                    wh = torch.sigmoid(wh)

                boxes, clss, scores = hm2box(heatmap        = heatmap.squeeze_(0),
                                             offset         = offset.squeeze_(0),
                                             wh             = wh.squeeze_(0),
                                             scale_factor   = self.scale_factor,
                                             topk           = self.topk,
                                             conf_th        = self.conf_th,
                                             normalized     = self.normalized_coord)
                stack_boxes.append(boxes)
                stack_clss.append(clss)
                stack_scores.append(scores)

            boxes = torch.cat(stack_boxes, dim=0)
            clss = torch.cat(stack_clss, dim=0)
            scores = torch.cat(stack_scores, dim=0)

            # non maximum suppression
            boxes, clss, scores = self.nms(boxes, clss, scores, self.nms_th)

            return boxes, clss, scores

    def nms(self, boxes, clss, scores, threshold):
        '''
            boxes: tensor (N, 4)
            clss: tensor (N)
            scores: tensor (N)
            threshold: float
        '''
        unique_indices = nms_pytorch(boxes, scores, threshold)

        return boxes[unique_indices], clss[unique_indices], scores[unique_indices]

@torch.jit.script
def nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    indices = torch.argsort(scores, descending=True)
    _boxes = boxes[indices]
    _scores = scores[indices]

    for i in range(_boxes.shape[0]-1):
        if _scores[i] == 0:
            continue
        xmin, ymin, xmax, ymax = torch.split(_boxes[i], 1, 0)

        _xmin, _ymin, _xmax, _ymax = torch.split(_boxes[i+1:], 1, 1)

        # intersection area
        x1 = torch.max(xmin, _xmin)
        y1 = torch.max(ymin, _ymin)
        x2 = torch.min(xmax, _xmax)
        y2 = torch.min(ymax, _ymax)
        w = torch.clamp((x2 - x1 + 1), min=0)
        h = torch.clamp((y2 - y1 + 1), min=0)

        area = (xmax - xmin + 1) * (ymax - ymin + 1)
        _area = (_xmax - _xmin + 1) * (_ymax - _ymin + 1)
        overlap = w * h

        iou = overlap / (area + _area - overlap)

        _scores[i+1:] = _scores[i+1:] * (iou.squeeze() < threshold).float()

    return indices[_scores>0].long()

if __name__ == '__main__':
    from train import load_network
    from config import build_parser

    do_test = False # for debug
    device = torch.device('cpu')

    args = build_parser()

    # load network
    network, _, _, _ = load_network(args, device)
    # perdict
    #pre = Preprocess()
    predictor = Export(network          = network,
                       topk             = args.topk,
                       scale_factor     = args.scale_factor,
                       conf_th          = args.conf_th,
                       nms_th           = args.nms_th,
                       normalized_coord = args.normalized_coord).to(device)
    predictor.eval()

    ##################### model save at cpu #####################
    x = torch.randn(1, 3, 512, 512)
    traced_model_cpu = torch.jit.trace(predictor.cpu(), x.cpu())
    torch.jit.save(traced_model_cpu, "jit_traced_model_cpu.pth")
    print("Model saved at cpu")

    ##################### model save at gpu #####################
    x = torch.randn(1, 3, 512, 512)
    traced_model_cpu = torch.jit.trace(predictor.cuda(), x.cuda())
    torch.jit.save(traced_model_cpu, "jit_traced_model_gpu.pth")
    print("Model saved at gpu")

    if do_test:
        import cv2
        normalizer = get_normalizer(pretrained=args.pretrained)
        x = cv2.cvtColor(cv2.imread('../0.jpg'), cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, dsize=(512, 512), interpolation=cv2.INTER_AREA)
        x = torch.tensor(x) # x: HxWxC, 0.0 ~ 255.0
        x = x.permute(2, 0, 1)/255.0
        x = normalizer(x).unsqueeze(0)

        box_lst, cls_lst, score_lst = predictor(x.to(device))
        for i in range(box_lst.shape[0]):
            print(', '.join(map(str, box_lst[i].tolist())), ',', cls_lst[i].item(), ',',  score_lst[i].item())

        ############ check the output of python and traced models ################
        x = torch.ones(1, 3, 512, 512)
        box_lst, cls_lst, score_lst = predictor(x.to(device))

        traced_model = torch.jit.trace(predictor, torch.randn(1, 3, 512, 512))
        x = torch.ones(1, 3, 512, 512)
        box_lst2, cls_lst2, score_lst2 = traced_model(x)
        print('output python == output jit: ', torch.all(torch.eq(box_lst, box_lst2)))
