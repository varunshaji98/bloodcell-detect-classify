import os
import numpy as np
import cv2
import torch
import glob as glob
from collections import defaultdict
from lxml.etree import Element, SubElement, ElementTree

from model import create_model

def write_xml(folder, filename, bbox_list):

    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    # SubElement(root, 'path').text = './images' +  filename

    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'coco'

    # Details from first entry
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = '256'
    SubElement(size, 'height').text = '256'
    SubElement(size, 'depth').text = '3'
    SubElement(root, 'segmented').text = '0'

    for entry in bbox_list:
        
        [e_bbox, e_score, e_class_name] = entry  

        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = e_class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(e_bbox[0])
        SubElement(bbox, 'ymin').text = str(e_bbox[1])
        SubElement(bbox, 'xmax').text = str(e_bbox[2])
        SubElement(bbox, 'ymax').text = str(e_bbox[3])

        SubElement(obj, 'score').text = str(e_score)

    tree = ElementTree(root)  
    xml_filename = os.path.join('.', folder, os.path.splitext(filename)[0] + '.xml')
    tree.write(xml_filename, pretty_print=True)


# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=3).to(device)
model.load_state_dict(torch.load(
    '../outputs/model20.pth', map_location=device
))
model.eval()

# directory where all the images are present
DIR_TEST = '../data/test'
DIR_OUT = '../test_predictions'
test_images = glob.glob(f"{DIR_TEST}/*.png")
print(f"Test instances: {len(test_images)}")

# classes: 0 index is reserved for background
CLASSES = [
    'background', 'rbc', 'wbc'
]

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.75

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float64)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float)
    # add batch dimension
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        outputs = model(image)

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]


        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]

        # For xml output
        entries_by_image = []
        for _bbox, _score, _pred_class in zip(boxes, scores, pred_classes):
            entries_by_image.append([_bbox, _score, _pred_class])

        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 1)
            cv2.putText(orig_image, pred_classes[j], 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 
                        1, lineType=cv2.LINE_AA)

        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
        cv2.imwrite(f"{DIR_OUT}/{image_name}.png", orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)

    # Write detections to XML file
    write_xml(DIR_OUT, f"{image_name}.xml", entries_by_image)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()