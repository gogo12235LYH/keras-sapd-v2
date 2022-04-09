# import numpy as np


""" Dataset Setting """
_dataset_path = {
    'VOC': r'../VOCdevkit/VOC2012+2007',
    'DPCB': r'C:/works/DeepPCB_voc',
    'rPCB': r'../PCB_DATASET_Random_Crop'
}
DATASET = 'DPCB'
DATABASE_PATH = _dataset_path[DATASET]
# DATABASE_PATH = r'../VOCdevkit/VOC2012+2007'
# DATABASE_PATH = r'../PCB_DATASET_Random_Crop'
# DATABASE_PATH = r'../DeepPCB_voc'
# DATABASE_PATH = r'../MixPCB'
# DATABASE_PATH = r'../MixPCB_MixUp_Fix'

""" Hyper-parameter setting """
DB_MODE = 'tf'          # 'tf' or 'keras', it means that using tf.data or keras.util.sequence.
EPOCHs = 25
STEPs_PER_EPOCH = 4138  # steps in one epoch, DPCB:1000, VOC:16552
BATCH_SIZE = 4          # Global Batch size
NUM_CLS = 20
PHI = 1                 # B0:(512, 512), B1:(640, 640), B2:(768, 768), B3:(896, 896), B4:(1024, 1024) ~ B7(1048, 1048)
MULTI_GPU = 0
MIXED_PRECISION = 1

""" Optimizer Setting """
OPTIMIZER = 'AdamW'      # SGDW, AdamW, SGD, Adam
USE_NESTEROV = True     # For SGD and SGDW
BASE_LR = 4e-5        # SGDW: (2.5e-3), AdamW: (4e-5, 2.5e-6)
MIN_LR = 2.5e-6         # Adam, AdamW: (2.5e-6)
MOMENTUM = 0.9          # SGDW: (0.9)
DECAY = 1e-5

""" Callback Setting """
LR_Scheduler = 2        # 1 for Cosine Decay, 2 for Steps Decay
USING_HISTORY = 1       # IF Optimizer has weight decay and weight decay can be decayed, must set to be 1 or 2.
EARLY_STOPPING = 0
EVALUATION = 0          # AP.5, Return training and Inference model when creating model function is called.
TENSORBOARD = 0

""" Warm Up """
USING_WARMUP = 0
WP_EPOCHs = int(0.1 * EPOCHs)
WP_RATIO = 0.1

""" Cosine Decay learning rate scheduler setting """
# ALPHA = np.round(MIN_LR / BASE_LR, 4)
ALPHA = 0.              # Cosine Decay's alpha


""" Augmentation Setting """
MISC_AUG = 1
VISUAL_AUG = 0
MixUp_AUG = 0


""" Backbone: Feature Extractor """
BACKBONE_TYPE = 'ResNetV1'  # ResNetV1, SEResNet
BACKBONE = 50           # 50 101
FREEZE_BN = False
PRETRAIN = 1            # 0 for from scratch, 1 for Imagenet, 2 for the weight's path that you want to load in.
PRETRAIN_WEIGHT = './xxxx.h5'


""" Head: Subnetwork """
SHRINK_RATIO = .2      # Bounding Box shrunk ratio
HEAD = 'Std'          # 'Std', 'Mix', 'MP', '3MP'
SUBNET_DEPTH = 4        # Depth of Head Subnetworks


""" Neck: Feature Pyramid Network """


""" Model: Classification and Regression Loss """
IOU_LOSS = 'giou'       # Regression Loss: iou, giou, ciou, fciou
IOU_FACTOR = 1.0


""" Model Name: Date-Dataset-MixUp-HEAD-FSN-Optimizer """
DATE = '20220408-'
D_NAME = f'{DATASET}-'
H_NAME = f'H{HEAD}-'
O_NAME = f'{OPTIMIZER}'
T_NAME = f"E{EPOCHs}BS{BATCH_SIZE}B{PHI}R{BACKBONE}D{SUBNET_DEPTH}"
NAME = DATE + D_NAME + H_NAME + O_NAME + T_NAME


""" Model Detections: NMS, Proposal setting """
NMS = 1                 # 1 for NMS, 2 for Soft-NMS
NMS_TH = 0.5            # intersect of union threshold in same detections
SCORE_TH = 0.01         # the threshold of object's confidence score
DETECTIONS = 1000       # detecting proposals
