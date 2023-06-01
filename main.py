import argparse
import os
import pprint
import time
import torch

import torch.backends.cudnn as cudnn
from loguru import logger
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from model.config import cfg, update_config
from model.buol import BUOL
from utils import comm
from dataset import dataset, sampler
from utils.utils import AverageMeter, get_loss_info_str, to_device
from dataset.panoptic_reconstruction_quality import PanopticReconstructionQuality
from model.solver.build import build_optimizer, build_lr_scheduler
from model.solver.utils import get_lr_group_id


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(cfg, args)

    return args


def main():
    args = parse_args()
    logger.add(os.path.join(cfg.OUTPUT_DIR, '{time}.log'), format="{time} {level} {message}", level="INFO")
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    gpus = list(cfg.GPUS)
    distributed = len(gpus) > 1
    device = torch.device('cuda:{}'.format(args.local_rank))

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", rank=args.local_rank, world_size=len(gpus), #### worldneed fix
        )

    # build model
    model = BUOL(cfg.MODEL)
    logger.info("Model:\n{}".format(model))
    logger.info("Rank of current process: {}. World size: {}".format(comm.get_rank(), comm.get_world_size()))

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True
        )

    model_dict = model.module if distributed else model

    if cfg.MODEL.FREEZE2D:
        freeze_model = [model_dict.bb2d, model_dict.bu2d, model_dict.depth]
        for md in freeze_model:
            md.eval()
            for key, value in md.named_parameters():
                value.requires_grad = False

    ds = dataset(cfg.DATASET)
    if cfg.MODEL.EVAL:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler.InferenceSampler(len(ds)), cfg.IMS_PER_BATCH, drop_last=False)
        data_loader = torch.utils.data.DataLoader(
            ds,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler.TrainingSampler(len(ds), cfg.DATALOADER.TRAIN_SHUFFLE), cfg.IMS_PER_BATCH, drop_last=False)
        optimizer = build_optimizer(cfg, model_dict)
        lr_scheduler = build_lr_scheduler(cfg, optimizer)
        data_loader = torch.utils.data.DataLoader(
            ds,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
        )
        data_loader_iter = iter(data_loader)
        start_iter = 0
        max_iter = cfg.TRAIN.MAX_ITER
        best_param_group_id = get_lr_group_id(optimizer)

    # initialize model
    if os.path.isfile(cfg.MODEL.WEIGHTS):
        model_weights = torch.load(cfg.MODEL.WEIGHTS, map_location='cpu')
        if 'start_iter' in model_weights:
            model_weights = model_weights['state_dict']
        model_dict.load_state_dict(model_weights, strict=True)
        logger.info('Pre-trained model from {}'.format(cfg.MODEL.WEIGHTS))
    else:
        logger.info('No pre-trained weights, training from scratch.')

    # load model
    if cfg.TRAIN.RESUME:
        model_state_file = os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location='cpu')
            start_iter = checkpoint['start_iter']
            model_dict.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info('Loaded checkpoint (starting from iter {})'.format(checkpoint['start_iter']))

    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = {'Loss': AverageMeter()}
    metric = PanopticReconstructionQuality(
        matching_threshold=0.25,
        dataset=cfg.DATASET.DATASET,
        label_divisor=cfg.MODEL.POST_PROCESSING.LABEL_DIVISOR)

    if cfg.MODEL.EVAL:
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                data = to_device(data, device)
                image = data.pop('image')
                pred, loss = model(image, data)

                metric.add(pred, data)
                # Minkowski Engine recommendation
                torch.cuda.empty_cache()

                if i == 0 or (i + 1) % cfg.PRINT_FREQ == 0 or i + 1 == len(data_loader):  ##compute prq
                    log_res = ''
                    for key, val in metric.reduce().items():
                        if isinstance(val, int):
                            log_res += '{}: {}, '.format(key, val)
                        else:
                            log_res += '{}: {:.2f}, '.format(key, val * 100)
                    log_res = log_res[:-2]
                    logger.info('[{:5d}/{:5d}]\t'.format(i + 1, len(data_loader)) + log_res)
    else:
        for i in range(start_iter, max_iter):
            start_time = time.time()
            data = next(data_loader_iter)
            if not distributed:
                data = to_device(data, device)
            data_time.update(time.time() - start_time)

            image = data.pop('image')
            pred, loss = model(image, data)

            loss_all = sum(loss.values())
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            # Get lr.
            lr = optimizer.param_groups[best_param_group_id]["lr"]
            lr_scheduler.step()

            # Minkowski Engine recommendation
            torch.cuda.empty_cache()

            batch_time.update(time.time() - start_time)
            batch = image["color"].size(0)
            loss_meter['Loss'].update(loss_all.detach().cpu().item(), batch)
            for li in loss:
                if li not in loss_meter:
                    loss_meter[li] = AverageMeter()
                loss_meter[li].update(loss[li].detach().cpu().item(), batch)

            if comm.is_main_process():
                if i == 0 or (i + 1) % cfg.PRINT_FREQ == 0:
                    msg = '[{0}/{1}] LR: {2:.7f}\t' \
                          'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                            i + 1, max_iter, lr, batch_time=batch_time, data_time=data_time)
                    msg += get_loss_info_str(loss_meter)
                    logger.info(msg)
                if i == 0 or (i + 1) % cfg.CKPT_FREQ == 0:
                    if comm.is_main_process():
                        save_dict = {
                            'start_iter': i + 1,
                            'state_dict': model_dict.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                        }
                        torch.save(save_dict, os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar'))
                        torch.save(model_dict.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, 'state_{:0>6d}.pth'.format(i + 1)))


if __name__ == '__main__':
    main()
