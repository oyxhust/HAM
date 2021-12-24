import os
import os.path as osp
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
import time
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

# bbox_classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

class BBoxToMask(object):
    def __call__(self, bboxs, classes, img, im_w, im_h, name, using_crf):
        img = np.array(img)
        img = np.ascontiguousarray(img)
        mask = []
        for idx in range(len(classes)):
            class_gts = np.zeros((im_h, im_w))
            if classes[idx] in bboxs:
                mask_h = 0
                mask_w = 0
                j = 0
                for loc in bboxs[classes[idx]]:
                    mask_h += loc[2]
                    mask_w += loc[3]
                    class_gts[int(loc[1]):int(loc[1]+loc[3]), int(loc[0]):int(loc[0]+loc[2])] = 1
                    j += 1
                if using_crf:
                    if name == 'train':
                        mask_h /= float(j)
                        mask_w /= float(j)
                        mask_scale = min(mask_h, mask_w)
                        if mask_scale < 45:
                            s1 = 5
                            s2 = 2
                        elif mask_scale < 70:
                            s1 = 12
                            s2 = 5
                        elif mask_scale < 140:
                            s1 = 15
                            s2 = 6
                        elif mask_scale < 210:
                            s1 = 25
                            s2 = 11
                        elif mask_scale < 380:
                            s1 = 50
                            s2 = 13
                        else:
                            s1 = 110
                            s2 = 26
                        d = dcrf.DenseCRF2D(im_h, im_w, 2)
                        class_crf = class_gts.astype(np.uint32)
                        U = unary_from_labels(class_crf, 2, gt_prob=0.7, zero_unsure=False)
                        d.setUnaryEnergy(U)
                        pairwise_bilateral = create_pairwise_bilateral(sdims=(s1, s1), schan=(s2, s2, s2), img=np.expand_dims(img, -1), chdim=2)
                        d.addPairwiseEnergy(pairwise_bilateral, compat=10)
                        q = d.inference(5)
                        class_crf = np.argmax(q, axis=0).reshape(im_h, im_w)
                        class_gts = class_gts.astype(np.uint8) * class_crf.astype(np.uint8)
                    else:
                        class_gts = class_gts.astype(np.uint8)
                else:
                    class_gts = class_gts.astype(np.uint8)
            else:
                class_gts = class_gts.astype(np.uint8)
            class_gts = Image.fromarray(class_gts)
            mask.append(class_gts)
        return mask


class NIHDataset(data.Dataset):
    def __init__(self, name, root, classes, image_sets='train.txt', helper='real_masks.txt', using_crf=False,
                 joint_transform=None, transform=None, mask_transform=BBoxToMask()):
        self.root = root
        self.name = name
        self.classes = classes
        self.image_set = image_sets
        self.helper = helper
        self.using_crf = using_crf
        self.transform = transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform
        self._imgpath = osp.join('%s', 'images', '%s')
        # classification label
        labelpath = os.path.join(self.root, 'Annotations', 'Tags.json')
        self.labels = json.load(open(labelpath))
        # segmentation label
        bboxpath = osp.join(self.root, 'Annotations', 'BBoxes.json')
        self.bboxs = json.load(open(bboxpath))
        self.ids = list()
        # print(self.name)
        if self.name in ['train', 'val-cls']:
            img_file = osp.join(self.root, 'ImageSets', 'Main', self.image_set)
        elif self.name == 'val-iou':
            img_file = osp.join(self.root, 'ImageSets', 'bbox', self.image_set)
        for line in open(img_file):
            self.ids.append((self.root, line.strip()))
        if self.name == 'train':
            self.helper_ids = list()
            for line in open(osp.join(self.root, 'ImageSets', 'bbox', self.helper)):
                self.helper_ids.append((self.root, line.strip()))


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # end = time.time()
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert('RGB')
        im_w, im_h = img.size

        cls_label = np.zeros(len(self.classes), dtype=np.float32)
        bbox_img_flag = False
        bbox_tags = np.zeros(len(self.classes), dtype=np.float32)
        for idx in range(len(self.classes)):
            if self.classes[idx] in self.labels[img_id[1]]:
                cls_label[idx] = 1
            # if self.classes[idx] in bbox_classes:
            if img_id[1] in self.bboxs:
                if self.classes[idx] in self.bboxs[img_id[1]]:
                    bbox_tags[idx] = 1
                    bbox_img_flag = True

        if bbox_img_flag==False:
            bbox_tags = cls_label

        # print('hehe')
        if img_id[1] in self.bboxs:
            mask = self.mask_transform(self.bboxs[img_id[1]], self.classes, img, im_w, im_h, self.name, self.using_crf)
        else:
            mask = []
            for idx in range(len(self.classes)):
                class_mask = np.zeros((im_h, im_w), dtype=np.uint8)
                class_mask = Image.fromarray(class_mask)
                mask.append(class_mask)

        flag = np.zeros(1, dtype=np.float32)
        if self.name == 'train':
            flag[0] = 0
            if img_id in self.helper_ids:
                flag[0] = 1
        elif self.name == 'val-cls' or 'val-iou' or 'test-cls' or 'test-iou':
            flag[0] = 1

        if self.joint_transform is not None:
            out_list = self.joint_transform([img]+mask)
            img = out_list[0]
            mask = out_list[1:]

        if self.transform is not None:
            img = self.transform(img)

        label = []
        for idx in range(len(self.classes)):
            class_gt = np.array(mask[idx])
            class_gt = torch.from_numpy(np.array(class_gt))
            label.append(class_gt.unsqueeze(0))
        mask = torch.cat(label, dim=0).float()
 
        cls_labels = torch.from_numpy(cls_label).float()
        flag = torch.from_numpy(flag).float()
        bbox_tags = torch.from_numpy(bbox_tags).float()
        # print(time.time() - end)
        
        return img, mask, cls_labels, flag, bbox_tags