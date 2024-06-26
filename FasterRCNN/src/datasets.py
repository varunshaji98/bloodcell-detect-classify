import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et
from config import MODEL_TYPE, CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, collate_fn_YOLO, get_train_transform, get_valid_transform, get_train_transform_YOLO, get_valid_transform_YOLO
from utils import horisontal_flip

# the dataset class
class MicrocontrollerDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        
        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.png")
        self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        if MODEL_TYPE == 'FasterRCNN':
            # capture the corresponding XML file for getting the annotations
            annot_filename = image_name[:-4] + '.xml'
            annot_file_path = os.path.join(self.dir_path, annot_filename)
            
            boxes = []
            labels = []
            tree = et.parse(annot_file_path)
            root = tree.getroot()
            
            # get the height and width of the image
            image_width = image.shape[1]
            image_height = image.shape[0]
            
            # box coordinates for xml files are extracted and corrected for image size given
            for member in root.findall('object'):
                # map the current object name to `classes` list to get...
                # ... the label index and append to `labels` list
                labels.append(self.classes.index(member.find('name').text))
                
                # xmin = left corner x-coordinates
                xmin = int(float(member.find('bndbox').find('xmin').text))
                # xmax = right corner x-coordinates
                xmax = int(float(member.find('bndbox').find('xmax').text))
                # ymin = left corner y-coordinates
                ymin = int(float(member.find('bndbox').find('ymin').text))
                # ymax = right corner y-coordinates
                ymax = int(float(member.find('bndbox').find('ymax').text))
                
                # resize the bounding boxes according to the...
                # ... desired `width`, `height`
                xmin_final = (xmin/image_width)*self.width
                xmax_final = (xmax/image_width)*self.width
                ymin_final = (ymin/image_height)*self.height
                yamx_final = (ymax/image_height)*self.height
                
                boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
            
            # bounding box to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # area of the bounding boxes
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # no crowd instances
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            # labels to tensor
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # prepare the final `target` dictionary
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            image_id = torch.tensor([idx])
            target["image_id"] = image_id

            # apply the image transforms
            if self.transforms:
                sample = self.transforms(image = image_resized,
                                        bboxes = target['boxes'],
                                        labels = labels)
                image_resized = sample['image']
                target['boxes'] = torch.Tensor(sample['bboxes'])
                
            return image_resized, target
        
        elif MODEL_TYPE == 'YOLO':
            # capture the corresponding XML file for getting the annotations
            annot_filename = image_name[:-4] + '.txt'
            annot_file_path = os.path.join(self.dir_path, annot_filename)
            
            boxes = []
            labels = []
            target = []
            
            # get the height and width of the image
            image_width = image.shape[1]
            image_height = image.shape[0]

            annotations = np.loadtxt(annot_file_path).reshape(-1, 5)
            # labels = annotation[:, 0]
            # boxes = annotation[:, 1:]

            # apply the image transforms
            if self.transforms:
                image_resized = torch.Tensor(image_resized)
                image_resized = image_resized.permute(2, 0, 1)
                ## Albumentations doesnt work properly for YOLO
                if np.random.random() < 0.5 and self.transforms == 'TRAIN':
                    image_resized, annotations  = horisontal_flip(image_resized, annotations)
                # Covert annotation target to format [IMG_INDEX, CLASS, X, Y, H, W]
                target = torch.zeros((len(annotations), 6))
                target[:, 1:] = torch.Tensor(annotations)

            return image_resized, target
        else:
            print('[ERROR] Unknown Model Type (should be FasterRCNN or YOLO)')
            return -1

    def __len__(self):
        return len(self.all_images)

# prepare the final datasets and data loaders
if MODEL_TYPE == 'FasterRCNN':
    train_dataset = MicrocontrollerDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
    valid_dataset = MicrocontrollerDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
elif MODEL_TYPE == 'YOLO':
    train_dataset = MicrocontrollerDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, 'TRAIN')
    valid_dataset = MicrocontrollerDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, 'VALID')
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_YOLO
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_YOLO
    )
else:
    print('[ERROR] Unknown Model Type (should be FasterRCNN or YOLO)')
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    # dataset = MicrocontrollerDataset(
    #     TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform_YOLO()
    # )
    dataset = MicrocontrollerDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, 'VALID'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_YOLO
    )
        # Display image and label.
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        box = target['boxes'][0]
        label = CLASSES[target['labels'][0]]
        cv2.rectangle(
            image, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 2
        )
        cv2.putText(
            image, label, (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )
        cv2.imshow('Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        
    # NUM_SAMPLES_TO_VISUALIZE = 5
    # for i in range(NUM_SAMPLES_TO_VISUALIZE):
    #     image, target = dataset[i]
    #     visualize_sample(image, target)