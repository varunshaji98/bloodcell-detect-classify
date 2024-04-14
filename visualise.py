import os
import cv2
from xml.etree import ElementTree as et
import albumentations as A


classes = [
    'background', 'rbc', 'wbc'
]
image_transform = False

image_name = "image-12.png"
img_file_path = os.path.join('images', image_name)
annot_filename = image_name[:-4] + '.xml'
annot_file_path = os.path.join('image_xmls', annot_filename)

boxes = []
labels = []
tree = et.parse(annot_file_path)
root = tree.getroot()

image = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
# get the height and width of the image
image_width = image.shape[1]
image_height = image.shape[0]

# box coordinates for xml files are extracted and corrected for image size given
for member in root.findall('object'):
        # map the current object name to `classes` list to get...
        # ... the label index and append to `labels` list
        labels.append(classes.index(member.find('name').text))
        
        # xmin = left corner x-coordinates
        xmin = int(float(member.find('bndbox').find('xmin').text))
        # xmax = right corner x-coordinates
        xmax = int(float(member.find('bndbox').find('xmax').text))
        # ymin = left corner y-coordinates
        ymin = int(float(member.find('bndbox').find('ymin').text))
        # ymax = right corner y-coordinates
        ymax = int(float(member.find('bndbox').find('ymax').text))
        
        boxes.append([xmin, ymin, xmax, ymax])

if image_transform:
    transform = A.Compose([
                    A.Flip(0.5),
                    A.RandomRotate90(0.5),
        ], bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels']
        })

    transformed = transform(image=image, bboxes=boxes, labels=labels)
    image = transformed['image']
    boxes = transformed['bboxes']
    labels = transformed['labels']

print(zip(boxes, labels))
for box, label in zip(boxes, labels):
        cv2.rectangle(image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 0, 255), 1)
        cv2.putText(image, classes[label], 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 
                    1, lineType=cv2.LINE_AA)

cv2.imshow('Image Sample', image)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image