import argparse
import pprint
import torch
import numpy
import marching_cubes as mcubes
import os

import torch.backends.cudnn as cudnn
from PIL import Image
from loguru import logger
from model.config import cfg, update_config
from model.buol import BUOL
from utils.intrinsics import adjust_intrinsic
from utils import transforms2d as t2d
from utils.io import write_ply


def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='configs/front.yaml',
                        type=str)
    parser.add_argument("--image", type=str, default='demo/demo.png')
    parser.add_argument('--intrinsic', type=str, default='demo/demo.txt')
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
    device = torch.device('cuda:0')

    # build model
    model = BUOL(cfg.MODEL)
    logger.info("Model:\n{}".format(model))

    model = model.to(device)
    model_dict = model

    # load model
    assert os.path.isfile(cfg.MODEL.WEIGHTS)
    model_weights = torch.load(cfg.MODEL.WEIGHTS, map_location='cpu')
    if 'start_iter' in model_weights:
        model_weights = model_weights['state_dict']
    model_dict.load_state_dict(model_weights, strict=True)
    logger.info('Pre-trained model from {}'.format(cfg.MODEL.WEIGHTS))

    transforms = t2d.Compose([
        t2d.ToTensor(),
        t2d.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(args.image)
    image = image.resize([320, 240])
    image = torch.tensor(transforms(image), device=device)[None]

    intrinsic = []
    with open(args.intrinsic) as f:
        for li in f.readlines():
            intrinsic.append([float(vi) for vi in li.strip().split(',')])
    intrinsic = adjust_intrinsic(intrinsic, [240, 320], [120, 160])
    intrinsic = torch.tensor(intrinsic, device=device, dtype=torch.float32)[None]

    model.to(device)
    model.eval()
    with torch.no_grad():
        image = {'color': image, 'intrinsic_matrix': intrinsic}
        pred = model(image)

        geometry = pred['geometry'][0, 0].detach().abs().cpu().numpy()
        panoptic = pred['panoptic3d'][0, 0].detach().cpu().numpy()

    truncation = cfg.MODEL.PROJECTION.TRUNCATION
    color = numpy.zeros([*geometry.shape, 3], dtype=int)
    for ins in numpy.unique(panoptic):
        if ins != 0:
            color[panoptic == ins] = (numpy.random.random([1, 3]) * 255).astype(numpy.int)
    geometry[panoptic == 0] = truncation

    verts, faces = mcubes.marching_cubes_color(geometry, color, 1., truncation)
    color = verts[..., 3:]
    verts = verts[..., :3]
    output_file = args.image.replace('.png', '.ply')
    write_ply(verts, color, faces, output_file=output_file)
    logger.info('success predict panoptic 3D scene at: ' + output_file)


if __name__ == '__main__':
    main()
