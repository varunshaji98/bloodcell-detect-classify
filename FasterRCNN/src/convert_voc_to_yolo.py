# based on https://gist.github.com/M-Younus/ceaf66e11a9c0f555b66a75d5b557465

import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join

classes = ['rbc', 'wbc']

def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.png'):
        image_list.append(filename)

    return image_list

def getLabelsInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.txt'):
        image_list.append(filename)

    return image_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = max((box[0] + box[1])/2.0 - 1, 0.0)
    y = max((box[2] + box[3])/2.0 - 1, 0.0)
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    out_file = open(dir_path + '/' + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



full_dir_path = '../data/valid'
image_paths = getLabelsInDir(full_dir_path)
print(image_paths)
import shutil
for file in image_paths:
    shutil.copy(file, '../../data/labels')
# out_file = open('valid.txt', 'w')
# for file in image_paths:
#     out_file.write(str(file)+'\n')
# # create label files
# for image_path in image_paths:
#     convert_annotation(full_dir_path, image_path)