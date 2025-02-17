import os
import glob
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
import SimpleITK as sitk
from dataloading.image_transforms import RandomAffine, ElasticTransform
from utils import *

class LobeTrainDataset(Dataset):
    def __init__(self, _config):
            self.image_dirs = sorted(
                glob.glob(os.path.join(_config["datasets_dir"], f'train/*.{_config["image_file_type"]}')),
                key=lambda x: int(x.split(f'train{os.sep}')[-1].split(f'.{_config["image_file_type"]}')[0])
            )
            self.label_dirs = sorted(
                glob.glob(os.path.join(_config["datasets_dir"], f'train{os.sep}label{os.sep}label_*.{_config["label_file_type"]}')),
                key=lambda x: int(x.split('label_')[-1].split(f'.{_config["label_file_type"]}')[0])
            )
            
            self.image_len = len(self.image_dirs)
            self.max_iter = _config["max_iter"]
            self.output_depth = 512
            self.read = True
            if self.read:
            # Read and preprocess images and labels
                self.images = []
                self.labels = []
                self.images_test = []
                self.labels_test = []
                for image_path, label_path in zip(self.image_dirs, self.label_dirs):
                    img_sitk = sitk.ReadImage(image_path)
                    lbl_sitk = sitk.ReadImage(label_path)

                    image = sitk.GetArrayFromImage(sitk.DICOMOrient(img_sitk, 'LPS'))
                    label = sitk.GetArrayFromImage(sitk.DICOMOrient(lbl_sitk, 'LPS'))

                    # Reshape to (C, H, W, D) for images and (H, W, D) for labels
                    image = np.transpose(image, (2, 1, 0))[np.newaxis, ...]  # Add channel dimension
                    label = np.transpose(label, (2, 1, 0))  # Transpose to (H, W, D)

                    resized_image, resized_label = data_resize(
                        image, label, self.output_depth, image_interpolation_order=3, label_interpolation_order=0
                    )
                    self.images.append(np.squeeze(resized_image, axis=0))
                    self.labels.append(resized_label)

    def __len__(self):
        return self.max_iter

    def __getitem__(self, idx):
        label = idx % 5 + 1

        img1_idx, img2_idx = random.sample(range(self.image_len), 2)
        while img1_idx == img2_idx:
            img1_idx, img2_idx = random.sample(range(self.image_len), 2)
        
        img, lbl = self.images[img1_idx], self.labels[img1_idx]
        sup_img, sup_lbl = self.images[img2_idx], self.labels[img2_idx]

        sample = {}
        len_img, len_sup_img = img.shape[0], sup_img.shape[0]
        
        scale_factor = len_sup_img / len_img
        slicer_idx = random.randint(0, len_img - 1)

        img, lbl = img[slicer_idx], lbl[slicer_idx]
        sup_img, sup_lbl = sup_img[int(slicer_idx * scale_factor)], sup_lbl[int(slicer_idx * scale_factor)]

        slicer_idx = float(slicer_idx / len_img)
        unique_lbl = [label for label in np.unique(lbl) if label != 0]
        unique_sup_lbl = [label for label in np.unique(sup_lbl) if label != 0]

        if label in unique_lbl and label in unique_sup_lbl:
            slicer_label = label
        else:
            return self.__getitem__(idx)
        
        slicer_label = label # np.random.choice(intersection)
        lobe_lbl = 1 * (lbl == slicer_label)
        lung_lbl = 1 * (lbl > 0)
        sup_lobe_lbl = 1 * (sup_lbl == slicer_label)
        sup_lung_lbl = 1 * (sup_lbl > 0)

        if np.random.random(1) > 0.5:
            img, lobe_lbl, lung_lbl = self.geom_transform(img, lobe_lbl, lung_lbl)
        else:
            sup_img, sup_lobe_lbl, sup_lung_lbl = self.geom_transform(sup_img, sup_lobe_lbl, sup_lung_lbl)

        img = np.stack([img] * 3, axis=0)  # (3, H, W)
        sup_img = np.stack([sup_img] * 3, axis=0)  # (3, H, W)

        if np.random.random(1) > 0.5:
            img = self.gamma_tansform(img)
        else:
            sup_img = self.gamma_tansform(sup_img)

        # Prepare output tensors
        img = img[None, ...]  # (1, 3, H, W)
        label = np.stack([lung_lbl[None, ...], lobe_lbl[None, ...]])  # (2, 1, H, W)
        sup_image = sup_img[None, None, ...]  # (1, 1, 3, H, W)
        sup_fg_lbl = np.stack([sup_lung_lbl[None, ...], sup_lobe_lbl[None, ...]])  # (2, 1, H, W)
        sup_bg_lbl = 1 - sup_fg_lbl  # (2, 1, H, W)

        sample = {
            'support_images': torch.from_numpy(sup_image).float(),
            'support_fg_labels': torch.from_numpy(sup_fg_lbl).float(),
            'support_bg_labels': torch.from_numpy(sup_bg_lbl).float(),
            'query_images': torch.from_numpy(img).float(),
            'query_labels': torch.from_numpy(label).float(),
            'slicer_idx': slicer_idx,
            'slicer_label': slicer_label
        }
        return sample

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * np.power(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def geom_transform(self, img, mask1, mask2):
        affine = {'rotate': 5, 'shift': (5, 5), 'shear': 5, 'scale': (0.9, 1.2)}
        alpha = 10
        sigma = 5
        order = 3

        tfx = []
        tfx.append(RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso', True),
                                     order=order))
        tfx.append(ElasticTransform(alpha, sigma))
        transform = deftfx.Compose(tfx)

        img = img.astype(np.float32) if img.dtype != np.float32 else img
        mask1 = mask1.astype(np.uint8)
        mask2 = mask2.astype(np.uint8)

        # Combine img, mask1, and mask2 into a single multi-channel array
        combined = np.concatenate((img[..., None], mask1[..., None], mask2[..., None]), axis=-1)
        
        # Apply transformations
        combined_transformed = transform(combined)

        # Separate the transformed components
        img = combined_transformed[..., :-2]
        mask1 = np.rint(combined_transformed[..., -2]).astype(np.uint8)
        mask2 = np.rint(combined_transformed[..., -1]).astype(np.uint8)

        return img.squeeze(), mask1, mask2


class LobeTestDataset(Dataset):
    def __init__(self, _config):
            self.image_dirs = sorted(
                glob.glob(os.path.join(_config["datasets_dir"], f'test/*.{_config["image_file_type"]}')),
                key=lambda x: int(x.split(f'test{os.sep}')[-1].split(f'.{_config["image_file_type"]}')[0])
            )
            self.label_dirs = sorted(
                glob.glob(os.path.join(_config["datasets_dir"], f'test{os.sep}label{os.sep}label_*.{_config["label_file_type"]}')),
                key=lambda x: int(x.split('label_')[-1].split(f'.{_config["label_file_type"]}')[0])
            )

            self.image_len = len(self.image_dirs)
            self.label = None
            self.output_depth = 512

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        img = self.image_dirs[idx]
        lbl = self.label_dirs[idx]

        sample = {}
        img_sitk = sitk.ReadImage(img)
        lbl_sitk = sitk.ReadImage(lbl)

        image = sitk.GetArrayFromImage(sitk.DICOMOrient(img_sitk, 'LPS'))
        label = sitk.GetArrayFromImage(sitk.DICOMOrient(lbl_sitk, 'LPS'))

        image = np.transpose(image, (2, 1, 0))[np.newaxis, ...]  # Add channel dimension
        label = np.transpose(label, (2, 1, 0))  # Transpose to (H, W, D)

        resized_image, resized_label = data_resize(
            image, label, self.output_depth, image_interpolation_order=3, label_interpolation_order=0
        )

        resized_image = np.squeeze(resized_image, axis=0)
        resized_image = np.stack([resized_image, resized_image, resized_image], axis=1) 

        sample['image'] = torch.from_numpy(image) # (H, 3, W , Z)
        sample['label'] = torch.from_numpy(label)
        sample['rs_image'] = torch.from_numpy(resized_image)
        sample['rs_label'] = torch.from_numpy(resized_label)
        sample['rs_len'] = image.shape[3]
        return sample
    

class Lobe_inference(Dataset):
    def __init__(self, _config):
            self.image_dirs = sorted(
                glob.glob(os.path.join(_config["datasets_dir"], f'/*.{_config["image_file_type"]}')))
            self.image_len = len(self.image_dirs)

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        img_path = self.image_dirs[idx]

        itk_image = sitk.ReadImage(img_path)
        spacing = list(itk_image.GetSpacing()) 
        origin = list(itk_image.GetOrigin()) 
        direction = list(itk_image.GetDirection()) 
        
        itk_image = sitk.GetArrayFromImage(sitk.DICOMOrient(itk_image, 'LPS'))
        image1 = np.transpose(itk_image, (2, 1, 0))
        image = image1[np.newaxis, ...]
        resized_image, resized_label = data_resize(
            image, image1, 512, image_interpolation_order=3, label_interpolation_order=0
        )
        resized_image = np.squeeze(resized_image, axis = 0)
        resized_image = np.stack([resized_image, resized_image, resized_image], axis=1)
        sample = {}
        sample['image'] = torch.from_numpy(resized_image)


        sample['filename'] = os.path.splitext(os.path.basename(img_path))[0]
        sample['direction'] = direction
        sample['spacing'] = spacing
        sample['origin'] = origin
        sample['rs_len'] = image.shape[3]
        return sample
    
