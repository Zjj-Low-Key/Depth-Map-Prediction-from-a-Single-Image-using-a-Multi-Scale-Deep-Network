import os
import random
from PIL import Image
import numpy as np
import mindspore
import mindspore.dataset.vision as vision
import mindspore.ops as ops

class NYUDataset:
    def __init__(self,mode='train'):
        
        self.mode = mode
        self.filename_list = self.read_txtfiles()
        if self.mode == 'train':
            self.data_dir = '/data/dataset/nyu/nyu2_sync/sync/'
        else:
            self.data_dir = '/data/dataset/nyu/bts_official_test/test/'
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.min_depth = 0.1
        self.max_depth = 10.0
        self.target_size = (240, 320)  # H, W
        self.output_size = (55, 74)    # for 1/4 resolution
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def read_txtfiles(self):
        if self.mode == 'train':
            with open('/data/local_userdata/zhujiajun/mindspore/nyudepthv2_train_files_with_gt.txt','r') as f:
                lines = f.readlines()
                lines = [l.strip().split(' ') for l in lines]
        else:
            with open('/data/local_userdata/zhujiajun/mindspore/nyu_eigen_test.txt','r') as f:
                lines = f.readlines()
                lines = [l.strip().split(' ') for l in lines]
        return lines
    
    def __len__(self):
        return len(self.filename_list)
    
    def __getitem__(self, index):
        
        if self.mode == 'train':
            image_path, depth_path, _ = self.filename_list[index]
        else:
            image_path, depth_path, focal = self.filename_list[index]
        
        image = Image.open(os.path.join(self.data_dir, image_path)).convert('RGB')
        depth = Image.open(os.path.join(self.data_dir, depth_path))

        # Resize to 320x240
        image = image.resize(self.target_size[::-1], Image.BILINEAR)
        depth = depth.resize(self.target_size[::-1], Image.NEAREST)
        image = np.asarray(image) / 255
        image = image.astype(np.uint8)
        depth = np.asarray(depth).astype(np.float32) / 1000.0  # to meters

        # Mask: remove min/max depths
        if self.mode == 'train':
            valid_mask = np.zeros_like(depth)
            valid_mask[45:472,43:608] = 1
            depth_mask = np.logical_and(depth > self.min_depth, depth < self.max_depth)
            valid_mask = np.logical_and(valid_mask, depth_mask)
        else:
            eval_mask = np.zeros_like(depth)
            eval_mask[45:471, 41:601] = 1
            depth_mask = np.logical_and(depth > self.min_depth, depth < self.max_depth)
            valid_mask = np.logical_and(eval_mask, depth_mask)

        if self.mode == 'train':
            # 1. Random scaling
            scale = np.random.uniform(1.0, 1.5)
            image = self.rescale(image, scale)
            depth = self.rescale(depth, scale)
            valid_mask = self.rescale(valid_mask, scale, nearest=True).astype(bool)
            depth = depth / scale  # adjust depth after scaling

            # 2. Random rotation
            angle = np.random.uniform(-5, 5)
            image = self.rotate(image, angle)
            depth = self.rotate(depth, angle, nearest=True)
            valid_mask = self.rotate(valid_mask, angle, nearest=True).astype(bool)

            # 3. Random crop (center 228x304)
            image, depth, valid_mask = self.random_crop(image, depth, valid_mask, h=228, w=304)

            # 4. Color jitter (brightness)
            factor = np.random.uniform(0.8, 1.2)
            image = image * factor
            image = np.clip(image, 0, 1)

            # 5. Horizontal flip
            if random.random() > 0.5:
                image = np.flip(image, axis=1)
                depth = np.flip(depth, axis=1)
                valid_mask = np.flip(valid_mask, axis=1)

        else:
            # center crop for test
            image, depth, valid_mask = self.center_crop(image, depth, valid_mask, h=228, w=304)

        # Normalize image
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1).astype(np.float32)  # CHW

        # Log depth, after safe clip
        depth = np.clip(depth, self.min_depth, self.max_depth)
        depth = np.log(depth)
        depth = self.resize_output(depth)
        valid_mask = self.resize_output(valid_mask).astype(bool)

        depth = np.expand_dims(depth, axis=0)
        valid_mask = np.expand_dims(valid_mask, axis=0)

        return image, depth, valid_mask

    def rescale(self, img, scale, nearest=False):
        h, w = img.shape[:2]
        out_size = (int(h * scale), int(w * scale))
        mode = Image.NEAREST if nearest else Image.BILINEAR
        return np.asarray(Image.fromarray(img).resize(out_size[::-1], resample=mode))

    def rotate(self, img, angle, nearest=False):
        mode = Image.NEAREST if nearest else Image.BILINEAR
        return np.asarray(Image.fromarray(img).rotate(angle, resample=mode))

    def random_crop(self, img, depth, mask, h, w):
        H, W = img.shape[:2]
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)
        return (img[y:y+h, x:x+w],
                depth[y:y+h, x:x+w],
                mask[y:y+h, x:x+w])

    def center_crop(self, img, depth, mask, h, w):
        H, W = img.shape[:2]
        x = (W - w) // 2
        y = (H - h) // 2
        return (img[y:y+h, x:x+w],
                depth[y:y+h, x:x+w],
                mask[y:y+h, x:x+w])

    def resize_output(self, arr):
        return np.asarray(Image.fromarray(arr).resize(self.output_size[::-1], Image.NEAREST))
    