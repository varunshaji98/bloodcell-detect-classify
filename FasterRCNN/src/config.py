import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 256 # resize the image for training and transforms
NUM_EPOCHS = 20 # number of epochs to train for

LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML/txt files directory
TRAIN_DIR = '../data/train'
# validation images and XML/txt files directory
VALID_DIR = '../data/valid'

'''For FasterRCNN'''
MODEL_TYPE = 'FasterRCNN'
# classes: 0 index is reserved for background
CLASSES = [
    'background', 'rbc', 'wbc'
]
NUM_CLASSES = 3
# location to save model and plots
OUT_DIR = '../outputs'

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs