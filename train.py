import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.cuda.amp import autocast, GradScaler

from hourglass import StackedHourglass
from loss import LossCalculator
from optim import get_optimizer
from data import load_dataset
from utils import AverageMeter, blend_heatmap

from torchsummary import summary


def distributed_device_train(args):
    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size

    mp.spawn(distributed_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    return None

def distributed_worker(device, ngpus_per_node, args):
    torch.cuda.set_device(device)
    cudnn.benchmark = True
    print('%s: Use GPU: %d for training'%(time.ctime(), args.gpu_no[device]))

    rank        = args.rank * ngpus_per_node + device
    batch_size  = int(args.batch_size / ngpus_per_node)
    num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    # init process for distributed training
    dist.init_process_group(backend     = args.dist_backend,
                            init_method = args.dist_url,
                            world_size  = args.world_size,
                            rank        = rank)

    # load network
    network, optimizer, scheduler, loss_calculator = load_network(args, device)
    if device == 0:
        summary(network, input_size=(3, 512, 512))

    # load dataset
    dataset     = load_dataset(args)
    sampler     = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader  = torch.utils.data.DataLoader(dataset       = dataset,
                                              batch_size    = batch_size,
                                              num_workers   = num_workers,
                                              pin_memory    = True,
                                              sampler       = sampler,
                                              collate_fn    = dataset.collate_fn)

    # gradient scaler for automatic mixed precision
    scaler = GradScaler() if args.amp else None

    # training
    for epoch in range(args.start_epoch, args.end_epoch):
        sampler.set_epoch(epoch)

        # train one epoch
        train_step(dataloader, network, loss_calculator, optimizer, scheduler, scaler, epoch, device, args)

        # adjust learning rate
        scheduler.step()

        # save network
        if rank % ngpus_per_node == 0:
            torch.save({'epoch': epoch+1,
                        'state_dict': network.module.state_dict() if hasattr(network, 'module') else network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict() if scaler is not None else None,
                        'loss_log': loss_calculator.log}, os.path.join(args.save_path, 'check_point_%d.pth'%(epoch+1)))

    return None

def train_step(dataloader, network, loss_calculator, optimizer, scheduler, scaler, epoch, device, args):
    time_logger = defaultdict(AverageMeter)

    network.train()

    tictoc = time.time()
    for iteration, (image, gt_heatmap, gt_offset, gt_size, gt_mask, gt_dict) in enumerate(dataloader, 1):
        time_logger['data'].update(time.time() - tictoc)

        # forward
        autocast_flag = True if scaler is not None else False
        with autocast(enabled=autocast_flag):
            tictoc = time.time()
            outputs = network(image.to(device))
            time_logger['forward'].update(time.time() - tictoc)

            ## calculate losses per scales
            tictoc = time.time()
            total_loss = 0
            for output in outputs.split(1, dim=1):
                output.squeeze_(1)
                pred_heatmap, pred_offset, pred_size = output.split([args.num_cls, 2, 2], dim=1)
                pred_heatmap = torch.sigmoid(pred_heatmap)
                if args.normalized_coord:
                    pred_offset = torch.sigmoid(pred_offset)
                    pred_size = torch.sigmoid(pred_size)

                _total_loss = loss_calculator(pred_heatmap,
                                             pred_offset,
                                             pred_size,
                                             gt_heatmap.to(device),
                                             gt_offset.to(device),
                                             gt_size.to(device),
                                             gt_mask.to(device))
                total_loss += _total_loss
            time_logger['loss'].update(time.time() - tictoc)

        # gradient accumulation
        optimizatoin_flag = (iteration % args.sub_divisions == 0) or (iteration == len(dataloader))

        # backward
        tictoc = time.time()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if optimizatoin_flag:
                scaler.step(optimizer)
                scaler.update()
        else:
            total_loss.backward()
            if optimizatoin_flag:
                optimizer.step()

        if optimizatoin_flag:
            optimizer.zero_grad()
        time_logger['backward'].update(time.time() - tictoc)

        # loging
        if (iteration % args.print_interval == 0) and (device == 0):
            loss_log = loss_calculator.get_log()
            _log  = '%s: Epoch [%2d/%2d]'%(time.ctime(), epoch, args.end_epoch)
            _log += ', Iteration [%4d/%4d]'%(iteration, len(dataloader))
            _log += ', Loss [%s]'%(loss_log)
            _log += ', Time(ms) [data: %6.2f'%(time_logger['data'].avg * 1000)
            _log += ', forward: %6.2f'%(time_logger['forward'].avg * 1000)
            _log += ', backward: %6.2f'%(time_logger['backward'].avg * 1000)
            _log += ', loss: %6.2f]'%(time_logger['loss'].avg * 1000)
            print(_log)

            # save blended image
            blended_pred = blend_heatmap(image[0], pred_heatmap[0], args.pretrained)
            blended_gt   = blend_heatmap(image[0], gt_heatmap[0], args.pretrained)
            blended_pred.save(os.path.join(args.save_path, 'training_log', 'training_pred.png'))
            blended_gt.save(os.path.join(args.save_path, 'training_log', 'training_gt.png'))

        tictoc = time.time()

    return None

def load_network(args, device):
    network = StackedHourglass(num_stack    = args.num_stack,
                               in_ch        = args.hourglass_inch,
                               out_ch       = args.num_cls+4,
                               increase_ch  = args.increase_ch,
                               activation   = args.activation,
                               pool         = args.pool,
                               neck_activation = args.neck_activation,
                               neck_pool       = args.neck_pool).to(device)

    if len(args.gpu_no) > 1 and args.train_flag:
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[device])

    optimizer, scheduler, loss_calculator = None, None, None
    if args.train_flag:
        optimizer, scheduler = get_optimizer(network       = network,
                                             lr            = args.lr,
                                             lr_milestone  = args.lr_milestone,
                                             lr_gamma      = args.lr_gamma)

        loss_calculator = LossCalculator(hm_weight      = args.hm_weight,
                                         offset_weight  = args.offset_weight,
                                         size_weight    = args.size_weight,
                                         focal_alpha    = args.focal_alpha,
                                         focal_beta     = args.focal_beta).to(device)

    if args.model_load:
        check_point = torch.load(args.model_load, map_location=device)
        network.load_state_dict(check_point['state_dict'])
        print('%s: Weights are loaded from %s'%(time.ctime(), args.model_load))

        if args.train_flag:
            optimizer.load_state_dict(check_point['optimizer'])
            loss_calculator.log = check_point['loss_log']
            if scheduler is not None:
                scheduler.load_state_dict(check_point['scheduler'])

    return network, optimizer, scheduler, loss_calculator
