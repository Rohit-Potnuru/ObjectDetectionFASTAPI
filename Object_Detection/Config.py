import torch


INPUT_DIR = "images/input/"
OUTPUT_DIR = "images/output/"

DEVICE = torch.device("cpu")




VOC_LABELS = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

LABEL_MAP = {k: v + 1 for v, k in enumerate(VOC_LABELS)}
LABEL_MAP['background'] = 0
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

DISTINCT_COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
LABEL_COLOR_MAP = {k: DISTINCT_COLORS[i] for i, k in enumerate(LABEL_MAP.keys())}
MODEL_FILE = 'model/checkpoint_ssd300130.pth.tar'
