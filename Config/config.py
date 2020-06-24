from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()

# Set the class name
__C.YOLO.CLASSES = "D:\\coursera\\YoLoSerirs\\data\\coco.names"
__C.YOLO.ANCHORS = "D:\\coursera\\YoLoSerirs\\data\\anchors\\yolov4_anchors.txt"
__C.YOLO.ANCHORS_V3 = "D:\\coursera\\YoLoSerirs\\data\\anchors\\yolov3_anchors.txt"
__C.YOLO.ANCHORS_TINY = "D:\\coursera\\YoLoSerirs\\data\\anchors\\basline_tiny_anchors.txt"
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.STRIDES_TINY = [16, 32]
__C.YOLO.XYSCALE = [1.2, 1.1, 1.05]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = "D:\\coursera\\YoLoSerirs\\data\\val2017_train.txt"
# __C.TRAIN.ANNOT_PATH = "D:\\coursera\\YoLoSerirs\\data\\voc2007_train.txt"
__C.TRAIN.BATCH_SIZE = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE = 416
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LR_INIT = 1e-3
__C.TRAIN.LR_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.FISRT_STAGE_EPOCHS = 20
__C.TRAIN.SECOND_STAGE_EPOCHS = 30

__C.TEST = edict()
__C.TEST.ANNOT_PATH = "D:\\coursera\\YoLoSerirs\\data\\val2017_test.txt"
# __C.TEST.ANNOT_PATH = "D:\\coursera\\YoLoSerirs\\data\\voc2007_test.txt"
__C.TEST.INPUT_SIZE = 416
__C.TEST.BATCH_SIZE = 1
__C.TEST.DATA_AUG = False