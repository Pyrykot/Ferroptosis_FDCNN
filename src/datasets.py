import torch
import cv2
import numpy as np
import os
import glob as glob
import json
import random
import matplotlib.pyplot as plt
import config
import utils
from config import CLASSES, TRAIN_DIR, TESTING_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from builtins import len

# the dataset class
class FasterRCNN_dataset(Dataset):
    def __init__(self, dir_path, classes, keywords, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.classes = classes
        self.crop_size = config.CHUNK_SIZE
        self.keywords_list = keywords


        self.image_paths = glob.glob(f"{self.dir_path}/*.png")
        self.image_paths = [os.path.normpath(image_path) for image_path in self.image_paths]
        self.all_paths = [os.path.split(image_path)[1] for image_path in self.image_paths]
        self.all_paths = sorted(self.all_paths)
        self.preloaded_images = self.preload_images()
        self.preload_boxes, self.preload_labels = self.preload_json_data()

    def preload_images(self):
        images = []
        for image_name in self.all_paths:
            if any(ext.lower() in image_name.lower() for ext in self.keywords_list):
                image_path = os.path.join(self.dir_path, image_name)
                image = cv2.imread(image_path, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
                images.append(image)
        return images

    def preload_json_data(self):
        all_box = []
        all_lab = []
        for image_name in self.all_paths:
            if any(ext.lower() in image_name.lower() for ext in self.keywords_list):
                annot_filename = image_name[:-4] + '.geojson'
                annot_file_path = os.path.join(self.dir_path, annot_filename)

                with open(annot_file_path, 'r') as f:
                    data = json.load(f)
                    data = data['features']
                boxes = []
                labels = []

                # (4)
                # 0-------1
                # |       |
                # |       |
                # 3-------2
                for y in data:
                    labels.append(self.classes.index(y['properties']['classification']['name']))
                    # xmin = left corner x-coordinates
                    xmin = int(y['geometry']['coordinates'][0][0][0])
                    # xmax = right corner x-coordinates
                    xmax = int(y['geometry']['coordinates'][0][1][0])
                    # ymin = left corner y-coordinates
                    ymin = int(y['geometry']['coordinates'][0][1][1])
                    # ymax = right corner y-coordinates
                    ymax = int(y['geometry']['coordinates'][0][3][1])
                    boxes.append([xmin, ymin, xmax, ymax])
                all_box.append(boxes)
                all_lab.append(labels)
        return all_box, all_lab

    def random_crop(self, idx):
        #  To ensure that atleast one cell is visible in the cropped area new
        #  idx is selected if the crop area was empty.
        while True:
            # Random crop coordinates
            x_rand = random.randint(0, self.preloaded_images[idx].shape[1] - self.crop_size)
            y_rand = random.randint(0, self.preloaded_images[idx].shape[0] - self.crop_size)

            x_max = x_rand + self.crop_size
            y_max = y_rand + self.crop_size
            crop = self.preloaded_images[idx][y_rand:y_max, x_rand:x_max, :]

            new_boxes = []
            new_labels = []
            # Repostions relevant bounding boxes. If more than 60 % of the box is visible,
            # then the box is cropped to the new image edges. If less than 60, then the box is discarded.
            # Random shift of +-1 pixel is applied and checked for possible edge breaks
            for i, box in enumerate(self.preload_boxes[idx]):
                new_box = [box[0], box[1], box[2], box[3]]
                x_overlap = (min(box[2], x_max) - max(box[0], x_rand)) / (box[2] - box[0])
                y_overlap = (min(box[3], y_max) - max(box[1], y_rand)) / (box[3] - box[1])
                if x_overlap == 1.0 and y_overlap == 1.0:
                    shift_prob = random.randint(1, 100)
                    x_rand_shift = 0
                    y_rand_shift = 0
                    if shift_prob > 70:
                        x_rand_shift = random.randint(-1, 1)
                        y_rand_shift = random.randint(-1, 1)
                        if box[0] - x_rand + x_rand_shift < 0:
                            x_rand_shift = 0
                        if box[1] + y_rand_shift < 0:
                            y_rand_shift = 0
                        if box[2] - x_rand + x_rand_shift > len(crop[0]):
                            x_rand_shift = 0
                        if box[3] + y_rand_shift > len(crop):
                            y_rand_shift = 0

                    new_box[0] = max(box[0] - x_rand + x_rand_shift, 0)
                    new_box[1] = max(box[1] - y_rand + y_rand_shift, 0)
                    new_box[2] = min(box[2] - x_rand + x_rand_shift, self.crop_size)
                    new_box[3] = min(box[3] - y_rand + y_rand_shift, self.crop_size)
                    new_boxes.append(new_box)
                    new_labels.append(self.preload_labels[idx][i])

                elif x_overlap > 0.6 and y_overlap > 0.6:

                    new_box[0] = max(box[0] - x_rand, 0)
                    new_box[1] = max(box[1] - y_rand, 0)
                    new_box[2] = min(box[2] - x_rand, self.crop_size)
                    new_box[3] = min(box[3] - y_rand, self.crop_size)
                    new_boxes.append(new_box)
                    new_labels.append(self.preload_labels[idx][i])

            if len(new_boxes) > 0:
                break
            else:
                if idx < len(self.preloaded_images) - 1:
                    idx = idx + 1
                else:
                    idx = idx - 1
        return new_boxes, new_labels, crop

    def __getitem__(self, idx):
        boxes, labels, image = self.random_crop(idx)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=image,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        return image, target

    def __len__(self):
        return len(self.preloaded_images)


def get_FCNN_trainloader(transforms, keywords):
    train_dataset = FasterRCNN_dataset(TRAIN_DIR, CLASSES, keywords, transforms)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=collate_fn
    )
    return train_loader


def get_FCNN_testloader(transforms, keywords):
    test_dataset = FasterRCNN_dataset(TESTING_DIR, CLASSES, keywords, transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    return test_loader

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    loader = get_FCNN_testloader(utils.get_train_transform(), ['ri'])
    for batch_idx, inputs in enumerate(loader):
        images = inputs[0]
        targets = inputs[1]
        for i, img in enumerate(images):
            boxes = targets[i]['boxes'].numpy()
            labels = targets[i]['labels'].numpy()
            np_img = img.numpy()
            utils.show_img_w_boxes(np_img, "f", boxes=boxes, labels=labels, scale_factor=4)
        break
    plt.show()

