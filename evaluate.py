import os
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torchvision

from data import load_dataset
from train import load_network
from utils import AverageMeter, save_pickle
from transform import hm2box

def single_device_evaluate(args):
    device = torch.device('cpu' if -1 in args.gpu_no else 'cuda')
    print('%s: Use %s for evaluation'%(time.ctime(), 'CPU' if -1 in args.gpu_no else 'GPU: %d'%args.gpu_no[0]))

    # load network
    network, _, _, _ = load_network(args, device)

    # perdict
    predictor = Prediction(network          = network,
                           topk             = args.topk,
                           scale_factor     = args.scale_factor,
                           conf_th          = args.conf_th,
                           nms              = args.nms,
                           nms_th           = args.nms_th,
                           normalized_coord = args.normalized_coord).to(device)

    # load dataset
    dataset    = load_dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset     = dataset,
                                             batch_size  = args.batch_size,
                                             shuffle     = False,
                                             num_workers = args.num_workers,
                                             collate_fn  = dataset.collate_fn)

    # evaluate
    predictions = evaluate_step(dataloader, predictor, device, args)

    # save the result
    save_pickle(os.path.join(args.save_path, 'prediction_results.pickle'), predictions)

    # make text file to measure mAP
    save_path = os.path.join(args.save_path, 'txt')
    os.makedirs(save_path, exist_ok=True)
    for filename, prediction in predictions.items():
        filename = os.path.splitext(filename)[0] + '.txt'
        np.savetxt(os.path.join(save_path, filename), prediction, fmt='%d %f %d %d %d %d')

    return None

def evaluate_step(dataloader, predictor, device, args):
    time_logger = defaultdict(AverageMeter)

    prediction_results = {}

    predictor.eval()

    tictoc = time.time()
    for image, gt_heatmap, gt_offset, gt_size, gt_mask, gt_dict in tqdm(dataloader):
        time_logger['data'].update(time.time() - tictoc)

        tictoc = time.time()
        box_lst, cls_lst, score_lst = predictor(image.to(device))
        time_logger['forward'].update(time.time() - tictoc)

        for b in range(image.shape[0]):
            boxes, clss, scores, gt_info = box_lst[b], cls_lst[b], score_lst[b], gt_dict[b]

            # resize to origin image size
            origin_size = int(gt_info['annotation']['size']['width']), int(gt_info['annotation']['size']['height'])
            resized_size = args.imsize, args.imsize
            _boxes = resize_box_to_original_scale(boxes.detach().cpu().numpy(), origin_size, resized_size)

            # make single array
            clss_np = clss.detach().cpu().numpy()[:, np.newaxis]
            scores_np = scores.detach().cpu().numpy()[:, np.newaxis]
            boxes_np = np.asarray(_boxes)

            # save prediction results
            if boxes_np.shape[0] != 0:
                pred = np.hstack([clss_np, scores_np, boxes_np])
            else:
                pred = np.zeros((0, 6))
            prediction_results[gt_info['annotation']['filename']] = pred

        tictoc = time.time()
    _log  = '%s: Evaluation'%(time.ctime())
    _log += ', Time(ms) [data: %6.2f'%(time_logger['data'].avg * 1000)
    _log += ', forward: %6.2f]'%(time_logger['forward'].avg * 1000)
    print(_log)

    return prediction_results

def resize_box_to_original_scale(boxes, original_size, transformed_size):
    origin_width, origin_height = original_size
    trans_width, trans_height   = transformed_size

    rw = origin_width / trans_width
    rh = origin_height / trans_height

    resized_boxes = []
    for xmin, ymin, xmax, ymax in boxes:
        resized_boxes.append([xmin * rw, ymin * rh, xmax * rw, ymax * rh])

    return resized_boxes

class Prediction(torch.nn.Module):
    def __init__(self, network, topk, scale_factor, conf_th, nms, nms_th, normalized_coord=False):
        super(Prediction, self).__init__()

        self.network          = network
        self.topk             = topk
        self.scale_factor     = scale_factor
        self.conf_th          = conf_th
        self.nms              = nms
        self.nms_th           = nms_th
        self.normalized_coord = normalized_coord

    def forward(self, x):
        ''' x: input tensor (b, c, h, w) '''
        box_lst, cls_lst, score_lst = [], [], []

        batch_output = self.network(x) # b, n, num_cls+4, h, w
        num_cls = batch_output.size(2) - 4
        # batch-wise
        for outputs in batch_output.split(1, dim=0):
            # stack(scale)-wise
            stack_boxes, stack_clss, stack_scores = [], [], []
            for output in outputs.split(1, dim=1):
                output.squeeze_(1)
                heatmap, offset, wh = output.split([num_cls,2,2], dim=1)
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

            # non maximum suppression
            boxes, clss, scores = self.nonmaximum_supression(torch.cat(stack_boxes,  dim=0),
                                                             torch.cat(stack_clss,   dim=0),
                                                             torch.cat(stack_scores, dim=0))

            # append boxes per batch
            box_lst.append(boxes)
            cls_lst.append(clss)
            score_lst.append(scores)

        return box_lst, cls_lst, score_lst

    def nonmaximum_supression(self, boxes, clss, scores):
        '''
            boxes: tensor (N, 4)
            clss: tensor (N)
            scores: tensor (N)
        '''
        if self.nms == 'nms':
            unique_indices = torchvision.ops.nms(boxes, scores, self.nms_th)

        elif self.nms == 'soft-nms':
            unique_indices = soft_nms_pytorch(boxes, scores, thresh=self.conf_th)

        else:
            raise NotImplementedError('Not expected nms algorithm: %s'%self.nms)

        return boxes[unique_indices], clss[unique_indices], scores[unique_indices]

def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """ Author: Richard Fang(github.com/DocF)
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[x1, y1, x2, y2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        xx1 = np.maximum(dets[i, 0].to("cpu").detach().numpy(), dets[pos:, 0].to("cpu").detach().numpy())
        yy1 = np.maximum(dets[i, 1].to("cpu").detach().numpy(), dets[pos:, 1].to("cpu").detach().numpy())
        xx2 = np.minimum(dets[i, 2].to("cpu").detach().numpy(), dets[pos:, 2].to("cpu").detach().numpy())
        yy2 = np.minimum(dets[i, 3].to("cpu").detach().numpy(), dets[pos:, 3].to("cpu").detach().numpy())

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].long()

    return keep

if __name__ == '__main__':
    from config import get_arguments
    from data import INDEX2CLASS, CLASS2COLOR
    from utils import imload, draw_box, write_text

    args = get_arguments()

    device = torch.device('cpu' if -1 in args.gpu_no else 'cuda')

    network, _, _, _ = load_network(args, device)
    predictor = Prediction(network          = network,
                           topk             = args.topk,
                           scale_factor     = args.scale_factor,
                           conf_th          = args.conf_th,
                           nms              = args.nms,
                           nms_th           = args.nms_th,
                           normalized_coord = args.normalized_coord).to(device)
    predictor.eval()

    # single image prediction
    img_ten, img_pil, origin_size = imload(args.data, args.pretrained, args.imsize)
    box_ten, cls_ten, score_ten = predictor(img_ten.to(device))
    box_lst, cls_lst, score_lst = box_ten[0].tolist(), cls_ten[0].tolist(), score_ten[0].tolist()

    # clamp outside image
    box_lst = [list(map(lambda x: max(0, min(x, args.imsize)), box)) for box in box_lst]

    # draw box, class and score per prediction
    for i, (box, cls, score) in enumerate(zip(box_lst, cls_lst, score_lst)):
        img_pil = draw_box(img_pil, box, color=CLASS2COLOR[cls])
        if args.fontsize > 0:
            text = '%s: %1.2f'%(INDEX2CLASS[cls], score)
            coord = [box[0], box[1]-args.fontsize]
            img_pil = write_text(img_pil, text, coord, fontsize=args.fontsize)

        # resize origin scale of image
        xmin, ymin, xmax, ymax = box
        xmin = xmin*origin_size[0]/args.imsize
        ymin = ymin*origin_size[1]/args.imsize
        xmax = xmax*origin_size[0]/args.imsize
        ymax = ymax*origin_size[0]/args.imsize

        print('%s: Index: %3d, Class: %7s, Score: %1.2f, Box: %4d, %4d, %4d, %4d'%(time.ctime(), i, INDEX2CLASS[cls], score, xmin, ymin, xmax, ymax))

    # resize to origin size and save the result image
    img_pil.resize(origin_size).save(os.path.join(args.save_path, 'image.png'))
