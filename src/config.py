import torch
import os

# TRAINING
BATCH_SIZE = 8
NUM_EPOCHS = 300  # number of epochs to train for
SAVE_PLOTS_EPOCH = 10  # save loss plots every n:th epoch
SAVE_MODEL_EPOCH = 150  # save after epochs
LEARNING_RATE = 0.009
WEIGHT_DECAY = 0.000001
OPTIM_MOMENTUM = 0.8
MILESTONES = [150, 200]
GAMMA = 0.1
KEYWORDS = ["Kar", "Oci", "Pf", "Ri", "To", "Fa"]
MODEL_NAME = "test.PTH"
CLASSES = [
    'background', 'Normal', 'Ferroptosis'  # Class names
]

# MODEL PARAMETERS
NUM_CLASSES = 3
TRAIN_LAYERS = 3
ANCHOR_SIZES = ((4,), (8,), (12,), (15,), (20,), (25,))
ASPECT_RATIOS = ((0.5, 1, 1.5),) * len(ANCHOR_SIZES)
IMG_MEAN = [0.50052514, 0.50052514, 0.50052514]
IMG_STD = [0.09143959, 0.09143959, 0.09143959]
UPSCALE = 450

# INFERENCE
IOU_TRESHOLD_inference = 0.2
CHUNK_SIZE = 100
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Allows running on CPU if GPU is not detected, this is heavily discouraged.

ROOT_DIR = "D:/CNN"
FILES_DIR = os.path.join(ROOT_DIR, '../files')
TRAIN_DIR = os.path.join(ROOT_DIR, '../files', 'FCNN_train')
TESTING_DIR = os.path.join(ROOT_DIR, '../files', 'FCNN_test')
VALID_DIR = os.path.join(ROOT_DIR, '../files', 'FCNN_valid')
OUT_DIR = os.path.join(ROOT_DIR, '../outputs')

platemap_default_name = "platemap.plateMap"  # Reads platemap format from Incucyte platemap editor software
WELL_IND = []

positive_label = 2
excel_name = "test.xlsx"
