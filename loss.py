import torch
import torch.nn as nn

from utils import load_pickle, save_pickle

class LossCalculator(nn.Module):
    def __init__(self, hm_weight, offset_weight, size_weight, focal_alpha, focal_beta):
        super(LossCalculator, self).__init__()
        self.log = {'hm': [], 'offset': [], 'size': [], 'total': []}

        self.l1_criterion    = NormedL1Loss()
        self.focal_criterion = FocalLoss(alpha=focal_alpha, beta=focal_beta)

        self.hm_weight      = hm_weight
        self.offset_weight  = offset_weight
        self.size_weight    = size_weight

    def forward(self, phm, poff, psize, ghm, goff, gsize, mask):
        hm_loss     = self.focal_criterion(phm, ghm, mask)
        offset_loss = self.l1_criterion(poff, goff, mask)
        size_loss   = self.l1_criterion(psize, gsize, mask)

        total_loss  = hm_loss     * self.hm_weight + \
                      offset_loss * self.offset_weight + \
                      size_loss   * self.size_weight

        self.log['hm'].append(hm_loss.item())
        self.log['offset'].append(offset_loss.item())
        self.log['size'].append(size_loss.item())
        self.log['total'].append(total_loss.item())

        return total_loss

    def get_log(self, length=100):
        log = []
        for key in ['hm', 'offset', 'size', 'total']:
            if len(self.log[key]) < length:
                length = len(self.log[key])
            log.append('%s: %5.2f'%(key, sum(self.log[key][-length:]) / length))
        return ', '.join(log)

class NormedL1Loss(nn.Module):
    def __init__(self):
        super(NormedL1Loss, self).__init__()

    def forward(self, pred, gt, mask):
        loss    = torch.abs(pred * mask - gt * mask)
        loss    = torch.sum(loss, dim=[1,2,3]).mean()
        num_pos = torch.sum(mask).clamp(1, 1e30)
        return loss / num_pos

class FocalLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, pred, gt, mask, eps=1e-7):
        neg_inds = torch.ones_like(mask) - mask

        neg_weights = torch.pow(1 - gt, self.beta)

        pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, self.alpha) * mask
        neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        pos_loss = pos_loss.sum(dim=[1,2,3]).mean()
        neg_loss = neg_loss.sum(dim=[1,2,3]).mean()
        num_pos  = mask.sum().clamp(1, 1e30)
        return -(pos_loss + neg_loss) / num_pos
