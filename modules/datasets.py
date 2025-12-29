import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle
import numpy as np
from torchvision import transforms



def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.mask_path = "/".join(self.image_dir.split("/")[:-3])
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        with open(args.label_path, 'rb') as f:
            self.labels = pickle.load(f)

        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['disease_detected'] = torch.tensor([tokenizer.token2idx[each] for each in self.examples[i]['disease_detected']])
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


def mic_trans(mask_generator,img):
    transform = transforms.Compose([
        transforms.ToTensor()  # 将图片转换为张量
    ])
    mic_image = transform(img)
    mic_image = mask_generator.mask_image(mic_image)
    image_numpy = mic_image.numpy()
    image_numpy = image_numpy[0]
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 255).astype(np.uint8)

    return Image.fromarray(image_numpy)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        mask_bone_1, mask_lung_1, mask_heart_1, mask_mediastinum_1 = [load_pickle("../XProNet-main/data/iu_xray_segmentation/{}/0_mask/{}_concat.pkl".format( image_id, each))for each in ["bone", "lung", "heart", "mediastinum"]]
        mask_bone_2, mask_lung_2, mask_heart_2, mask_mediastinum_2 = [load_pickle("../XProNet-main/data/iu_xray_segmentation/{}/1_mask/{}_concat.pkl".format( image_id, each))for each in ["bone", "lung", "heart", "mediastinum"]]

        array = image_id.split('-')
        modified_id = array[0]+'-'+array[1]
        label = torch.FloatTensor(self.labels[modified_id])
        if self.transform is not None:
            image_1, mask_bone_1, mask_lung_1, mask_heart_1, mask_mediastinum_1 = \
                self.transform(image_1, mask_bone_1, mask_lung_1, mask_heart_1, mask_mediastinum_1)
            image_2, mask_bone_2, mask_lung_2, mask_heart_2, mask_mediastinum_2 = \
                self.transform(image_2, mask_bone_2, mask_lung_2, mask_heart_2, mask_mediastinum_2)
        image = torch.stack((image_1, image_2), 0)
        image_mask_bone = torch.stack((mask_bone_1, mask_bone_2), 0)
        image_mask_lung = torch.stack((mask_lung_1, mask_lung_2), 0)
        image_mask_heart = torch.stack((mask_heart_1, mask_heart_2), 0)
        image_mask_mediastinum = torch.stack((mask_mediastinum_1, mask_mediastinum_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        disease_detected = example['disease_detected']
        disease_detected_len = len(example['disease_detected'])
        sample = (image_id, image, image_mask_bone, image_mask_lung, image_mask_heart, image_mask_mediastinum, report_ids, report_masks, seq_length, label, disease_detected, disease_detected_len)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        label = torch.FloatTensor(self.labels[example['id']])
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, label)
        return sample