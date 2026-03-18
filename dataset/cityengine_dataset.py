import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class CityEngineDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 256),
                 mean=(127.50044914,
                       128.43456556,
                       117.57694133), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        print("Processing Folder: ", self.root)
        print(f"Number of images in the DataLoader: {len(self.img_ids)}")
        self.item_number = len(self.img_ids)
        
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "imgs/%s" % name)
            label_file = osp.join(self.root, "masks/%s" % name.replace('rgb', 'seg'))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
            
    def __getNumItems__(self):
        return self.item_number

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        image = (image-128)
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), np.array(size), name


if __name__ == '__main__':
    FOLDER = 'building_dataset'
    SOURCE = 'OSU_building'
    ADAPTER = 'OSMOSU2DR_building'
    TARGET = 'OSU_building'
    SET = 'train'
    IMG_MEAN = np.array((128, 128, 128), dtype=np.float32)


    DATA_DIRECTORY_ADAPTER = f'./{FOLDER}/{ADAPTER}'
    DATA_LIST_PATH_ADAPTER = f'./{FOLDER}/{ADAPTER}/train.txt'
    
    dst = CityEngineDataSet(DATA_DIRECTORY_ADAPTER, DATA_LIST_PATH_ADAPTER, 
                                        max_iters=50000 * 1 * 8,
                                        crop_size='512,512', scale=False, mirror=True, mean=IMG_MEAN)
 
    print(f"////////////Loader: {dst.__getNumItems__}")