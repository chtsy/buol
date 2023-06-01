from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
# Logging frequency
_C.PRINT_FREQ = 20
# Checkpoint frequency
_C.CKPT_FREQ = 5000

# -----------------------------------------------------------------------------
# CUDNN
# -----------------------------------------------------------------------------
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.WEIGHTS = ''
# ---------------------------------------------------------------------------- #
# 2D model options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL2D = CN()
_C.MODEL.MODEL2D.BACKBONE = CN()

_C.MODEL.MODEL2D.BACKBONE.NAME = "resnet50"
# Controls output stride
_C.MODEL.MODEL2D.BACKBONE.DILATION = (False, False, False)
# pretrained backbone provided by official PyTorch modelzoo
_C.MODEL.MODEL2D.BACKBONE.PRETRAINED = True

# ---------------------------------------------------------------------------- #
# bottom-up model options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL2D.BOTTOM_UP = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.IN_FEATURE = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.IN_FEATURE.FEATURE_KEY = "res5"
_C.MODEL.MODEL2D.BOTTOM_UP.IN_FEATURE.IN_CHANNELS = 2048
_C.MODEL.MODEL2D.BOTTOM_UP.IN_FEATURE.LOW_LEVEL_KEY = ["res4", "res3", "res2"]
_C.MODEL.MODEL2D.BOTTOM_UP.IN_FEATURE.LOW_LEVEL_CHANNELS = (1024, 512, 256)

# ---------------------------------------------------------------------------- #
# semantic segmentation options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC.ASPP = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC.ASPP.ATROUS_RATES = (3, 6, 9)
_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC.ASPP.LOW_LEVEL_CHANNELS_PROJECT = (128, 64, 32)
_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC.ASPP.ASPP_CHANNELS = 256

_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC.DECODER = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC.DECODER.DECODER_CHANNELS = 256

_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC.HEAD = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC.HEAD.HEAD_CHANNELS = 64
_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC.HEAD.NUM_CLASSES = (19,)
_C.MODEL.MODEL2D.BOTTOM_UP.SEMANTIC.HEAD.CLASS_KEY = ["semantic"]

# ---------------------------------------------------------------------------- #
# semantic segmentation options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE.ASPP = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE.ASPP.ATROUS_RATES = (3, 6, 9)
_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE.ASPP.LOW_LEVEL_CHANNELS_PROJECT = (64, 32, 16)
_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE.ASPP.ASPP_CHANNELS = 256

_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE.DECODER = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE.DECODER.DECODER_CHANNELS = 128

_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE.HEAD = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE.HEAD.HEAD_CHANNELS = 32
_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE.HEAD.NUM_CLASSES = (1, 2)
_C.MODEL.MODEL2D.BOTTOM_UP.INSTANCE.HEAD.CLASS_KEY = ["center", "offset"]


_C.MODEL.MODEL2D.BOTTOM_UP.LOSS = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.LOSS.SEMANTIC = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.LOSS.SEMANTIC.NAME = 'CrossEntropyLoss'
_C.MODEL.MODEL2D.BOTTOM_UP.LOSS.SEMANTIC.WEIGHT = 1.
_C.MODEL.MODEL2D.BOTTOM_UP.LOSS.SEMANTIC.IGNORE = 255

_C.MODEL.MODEL2D.BOTTOM_UP.LOSS.CENTER = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.LOSS.CENTER.NAME = 'MSELoss'
_C.MODEL.MODEL2D.BOTTOM_UP.LOSS.CENTER.WEIGHT = 1.

_C.MODEL.MODEL2D.BOTTOM_UP.LOSS.OFFSET = CN()
_C.MODEL.MODEL2D.BOTTOM_UP.LOSS.OFFSET.NAME = 'L1Loss'
_C.MODEL.MODEL2D.BOTTOM_UP.LOSS.OFFSET.WEIGHT = 1.

# ---------------------------------------------------------------------------- #
# depth estimation options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL2D.DEPTH = CN()
_C.MODEL.MODEL2D.DEPTH.FEATURE_KEY = ["res5", "res4", "res3", "res2"]
_C.MODEL.MODEL2D.DEPTH.BLOCK_CHANNELS = (2048, 1024, 512, 256, 128)
_C.MODEL.MODEL2D.DEPTH.FEATURE_CHANNELS = 64
_C.MODEL.MODEL2D.DEPTH.NUM_CLASSES = (100, 1)
_C.MODEL.MODEL2D.DEPTH.CLASS_KEY = ["occupancy2d", "depth"]

_C.MODEL.MODEL2D.DEPTH.LOSS = CN()
_C.MODEL.MODEL2D.DEPTH.LOSS.DEPTH = CN()
_C.MODEL.MODEL2D.DEPTH.LOSS.DEPTH.WEIGHT = 10.
_C.MODEL.MODEL2D.DEPTH.LOSS.OCCUPANCY2D = CN()
_C.MODEL.MODEL2D.DEPTH.LOSS.OCCUPANCY2D.NAME = 'BCELoss'
_C.MODEL.MODEL2D.DEPTH.LOSS.OCCUPANCY2D.WEIGHT = 1.


# ---------------------------------------------------------------------------- #
# 3D model options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL3D = CN()

_C.MODEL.MODEL3D.BACKBONE = CN()

_C.MODEL.MODEL3D.BACKBONE.NAME = "resnet50"
# Controls output stride
_C.MODEL.MODEL3D.BACKBONE.DILATION = (False, False, False)
# pretrained backbone provided by official PyTorch modelzoo
_C.MODEL.MODEL3D.BACKBONE.PRETRAINED = True

# ---------------------------------------------------------------------------- #
# bottom-up model options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL3D.BOTTOM_UP = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.IN_FEATURE = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.IN_FEATURE.FEATURE_KEY = "res5"
_C.MODEL.MODEL3D.BOTTOM_UP.IN_FEATURE.IN_CHANNELS = 2048
_C.MODEL.MODEL3D.BOTTOM_UP.IN_FEATURE.LOW_LEVEL_KEY = ["res4", "res3", "res2"]
_C.MODEL.MODEL3D.BOTTOM_UP.IN_FEATURE.LOW_LEVEL_CHANNELS = (1024, 512, 256)

# ---------------------------------------------------------------------------- #
# semantic segmentation options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC.ASPP = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC.ASPP.ATROUS_RATES = (3, 6, 9)
_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC.ASPP.LOW_LEVEL_CHANNELS_PROJECT = (128, 64, 32)
_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC.ASPP.ASPP_CHANNELS = 256

_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC.DECODER = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC.DECODER.DECODER_CHANNELS = 256

_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC.HEAD = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC.HEAD.HEAD_CHANNELS = 64
_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC.HEAD.NUM_CLASSES = (19,)
_C.MODEL.MODEL3D.BOTTOM_UP.SEMANTIC.HEAD.CLASS_KEY = ["semantic"]

# ---------------------------------------------------------------------------- #
# instance segmentation options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE.ASPP = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE.ASPP.ATROUS_RATES = (3, 6, 9)
_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE.ASPP.LOW_LEVEL_CHANNELS_PROJECT = (64, 32, 16)
_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE.ASPP.ASPP_CHANNELS = 256

_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE.DECODER = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE.DECODER.DECODER_CHANNELS = 128

_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE.HEAD = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE.HEAD.HEAD_CHANNELS = 32
_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE.HEAD.NUM_CLASSES = (1, 2)
_C.MODEL.MODEL3D.BOTTOM_UP.INSTANCE.HEAD.CLASS_KEY = ["center", "offset"]

# ---------------------------------------------------------------------------- #
# geometry options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY.ASPP = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY.ASPP.ATROUS_RATES = (3, 6, 9)
_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY.ASPP.LOW_LEVEL_CHANNELS_PROJECT = (64, 32, 16)
_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY.ASPP.ASPP_CHANNELS = 256

_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY.DECODER = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY.DECODER.DECODER_CHANNELS = 128

_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY.HEAD = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY.HEAD.HEAD_CHANNELS = 32
_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY.HEAD.NUM_CLASSES = (1, 2)
_C.MODEL.MODEL3D.BOTTOM_UP.GEOMETRY.HEAD.CLASS_KEY = ["center", "offset"]

# ---------------------------------------------------------------------------- #
# 3d loss options
# ---------------------------------------------------------------------------- #
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.SEMANTIC = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.SEMANTIC.NAME = 'CrossEntropyLoss'
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.SEMANTIC.WEIGHT = 1.
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.SEMANTIC.IGNORE = 255

_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.OFFSET = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.OFFSET.NAME = 'L1Loss'
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.OFFSET.WEIGHT = 1.

_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.CENTER = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.CENTER.NAME = 'MSELoss'
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.CENTER.WEIGHT = 200.

_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.OCCUPANCY = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.OCCUPANCY.NAME = 'BCELoss'
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.OCCUPANCY.WEIGHT = 1.

_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.GEOMETRY = CN()
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.GEOMETRY.NAME = 'L1Loss'
_C.MODEL.MODEL3D.BOTTOM_UP.LOSS.GEOMETRY.WEIGHT = 1.


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.FILE_LIST = "resources/front3d/train_list_3d.txt"
_C.DATASET.FIELDS = ("color", "depth", "instance2d", "geometry", "instance3d", "semantic3d")
_C.DATASET.ROOT = "datasets/front3d/"
_C.DATASET.DATASET = "front3d"
_C.DATASET.IMAGE_SIZE = (320, 160)
_C.DATASET.INTRINSIC = [[ 277.1281435,   0.       , 159.5,  0.],
                        [   0.       , 277.1281435, 119.5,  0.],
                        [   0.       ,   0.       ,   1. ,  0.],
                        [   0.       ,   0.       ,   0. ,  1.]]  # Fixed 3D-Front intrinsic

_C.DATASET.NUM_CLASSES = 13
_C.DATASET.TRAIN_SPLIT = 'train'
_C.DATASET.TEST_SPLIT = 'val'
_C.DATASET.MIRROR = True
_C.DATASET.MIN_SCALE = 0.5
_C.DATASET.MAX_SCALE = 2.0
_C.DATASET.SCALE_STEP_SIZE = 0.1
_C.DATASET.MEAN = (0.485, 0.456, 0.406)
_C.DATASET.STD = (0.229, 0.224, 0.225)
_C.DATASET.IGNORE_STUFF_IN_OFFSET = True
_C.DATASET.SMALL_INSTANCE_AREA = 0
_C.DATASET.SMALL_INSTANCE_WEIGHT = 1

_C.DATASET.MIN_RESIZE_VALUE = -1
_C.DATASET.MAX_RESIZE_VALUE = -1
_C.DATASET.RESIZE_FACTOR = -1
_C.DATASET.STUFF_CLASSES = [0, 10, 11, 12]

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.WEIGHT_DECAY = 0.0001
# Weight decay of norm layers.
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
# Bias.
_C.SOLVER.BIAS_LR_FACTOR = 2.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.OPTIMIZER = 'sgd'
_C.SOLVER.ADAM_BETAS = (0.9, 0.999)
_C.SOLVER.ADAM_EPS = 1e-08

_C.SOLVER.LR_SCHEDULER_NAME = 'WarmupPolyLR'
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.POLY_LR_POWER = 0.9
_C.SOLVER.POLY_LR_CONSTANT_ENDING = 0

_C.SOLVER.CLIP_GRADIENTS = CN()
_C.SOLVER.CLIP_GRADIENTS.ENABLED = False
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.MAX_ITER = 90000
_C.TRAIN.RESUME = False

_C.IMS_PER_BATCH = 1
# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

_C.DATALOADER.SAMPLER_TRAIN = 'TrainingSampler'
_C.DATALOADER.TRAIN_SHUFFLE = True

_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# POST PROCESSING
# -----------------------------------------------------------------------------
_C.MODEL.POST_PROCESSING = CN()
_C.MODEL.POST_PROCESSING.CENTER_THRESHOLD = 0.1
_C.MODEL.POST_PROCESSING.NMS_KERNEL = 7
_C.MODEL.POST_PROCESSING.TOP_K_INSTANCE = 20 #200
_C.MODEL.POST_PROCESSING.STUFF_AREA = 64 #2048
_C.MODEL.POST_PROCESSING.LABEL_DIVISOR = 1000
_C.MODEL.POST_PROCESSING.THING_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9]

_C.MODEL.PROJECTION = CN()
_C.MODEL.PROJECTION.VOXEL_SIZE = 0.03
_C.MODEL.PROJECTION.DEPTH_MIN = 0.4
_C.MODEL.PROJECTION.DEPTH_MAX = 6.0
_C.MODEL.PROJECTION.INTRINSIC = [[277.1281435,   0.       , 159.5,  0.],
                                 [  0.       , 277.1281435, 119.5,  0.],
                                 [  0.       ,   0.       ,   1. ,  0.],
                                 [  0.       ,   0.       ,   0. ,  1.]]  # Fixed 3D-Front intrinsic

_C.MODEL.PROJECTION.GRID_DIMENSIONS = [256, 256, 256]
_C.MODEL.PROJECTION.IMAGE_SIZE = (160, 120)
_C.MODEL.PROJECTION.TRUNCATION = 3.0


_C.MODEL.EVAL = False
_C.MODEL.FREEZE2D = False


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    if cfg.MODEL.EVAL:
        cfg.DATASET.FIELDS = cfg.DATASET.FIELDS + ("panoptic", )

    cfg.DATASET.PROJECTION = cfg.MODEL.PROJECTION
    cfg.MODEL.MODEL2D.BOTTOM_UP.IMAGE_SIZE = cfg.MODEL.PROJECTION.IMAGE_SIZE
    cfg.MODEL.MODEL3D.BOTTOM_UP.GRID_DIMENSIONS = cfg.MODEL.PROJECTION.GRID_DIMENSIONS
    cfg.MODEL.MODEL3D.BOTTOM_UP.TRUNCATION = cfg.MODEL.PROJECTION.TRUNCATION
    cfg.MODEL.POST_PROCESSING.DATASET = cfg.DATASET.DATASET

    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
