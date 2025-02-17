import os
import random
import numpy as np
import SimpleITK as sitk
import pickle
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Scores():
    def __init__(self, num_classes, device="cuda"):
        self.num_classes = num_classes
        self.device = device  
        self.TP = torch.zeros(num_classes, dtype=torch.float32, device=self.device)
        self.TN = torch.zeros(num_classes, dtype=torch.float32, device=self.device)
        self.FP = torch.zeros(num_classes, dtype=torch.float32, device=self.device)
        self.FN = torch.zeros(num_classes, dtype=torch.float32, device=self.device)

        self.patient_dice = {label: [] for label in range(1, num_classes + 1)}
        self.patient_iou = {label: [] for label in range(1, num_classes + 1)}

    def record(self, preds, label):
        assert preds.shape == label.shape, "Shape of preds and label must match."

        sample_dice = {}
        sample_iou = {}

        for cls in range(1, self.num_classes + 1):
            cls_preds = (preds == cls)
            cls_label = (label == cls)

            tp = torch.sum((cls_label == 1) * (cls_preds == 1))
            tn = torch.sum((cls_label == 0) * (cls_preds == 0))
            fp = torch.sum((cls_label == 0) * (cls_preds == 1))
            fn = torch.sum((cls_label == 1) * (cls_preds == 0))

            dice = 2 * tp / (2 * tp + fp + fn + 1e-6)  
            iou = tp / (tp + fp + fn + 1e-6)

            dice_value = dice.item()
            iou_value = iou.item()

            self.patient_dice[cls].append(dice_value)
            self.patient_iou[cls].append(iou_value)

            self.TP[cls - 1] += tp
            self.TN[cls - 1] += tn
            self.FP[cls - 1] += fp
            self.FN[cls - 1] += fn

            sample_dice[cls] = dice_value
            sample_iou[cls] = iou_value

        print("-" * 30)
        for cls in range(1, self.num_classes + 1):
            print(f" Class {cls}: Dice = {sample_dice[cls]:.4f}, IoU = {sample_iou[cls]:.4f}")
        print("-" * 30)


    def compute_dice(self):
        """Compute the Dice score for each label (excluding label 0)."""
        dice_scores = 2 * self.TP / (2 * self.TP + self.FP + self.FN + 1e-6)
        return dice_scores

    def compute_iou(self):
        """Compute the IoU score for each label (excluding label 0)."""
        iou_scores = self.TP / (self.TP + self.FP + self.FN + 1e-6)
        return iou_scores

    def average_patient_dice(self):
        """Compute the average Dice score across all patients for each label (excluding label 0)."""
        avg_dice = {cls: sum(self.patient_dice[cls]) / len(self.patient_dice[cls]) if self.patient_dice[cls] else 0 for cls in range(1, self.num_classes + 1)}
        
        print("\n[Average Dice]")
        print("-" * 30)
        for cls in range(1, self.num_classes + 1):
            print(f" Class {cls}: {avg_dice[cls]:.4f}")
        print("-" * 30)
        
        return avg_dice

    def average_patient_iou(self):
        """Compute the average IoU score across all patients for each label (excluding label 0)."""
        avg_iou = {cls: sum(self.patient_iou[cls]) / len(self.patient_iou[cls]) if self.patient_iou[cls] else 0 for cls in range(1, self.num_classes + 1)}
        
        print("\n[Average IoU]")
        print("-" * 30)
        for cls in range(1, self.num_classes + 1):
            print(f" Class {cls}: {avg_iou[cls]:.4f}")
        print("-" * 30)

        return avg_iou
    


########################################################################################################################
def data_resize(image: np.ndarray, 
                        label: np.ndarray, 
                        output_depth: int, 
                        image_interpolation_order: int = 3, 
                        label_interpolation_order: int = 0):
    """
    image : np.ndarray
        The 3D image array with channels, shape (C, H, W, D).
    label : np.ndarray
        The 3D label array, shape (H, W, D).
    """

    assert len(image.shape) == 4, "Image must have 4 dimensions (C, H, W, D)."
    assert len(label.shape) == 3, "Label must have 3 dimensions (H, W, D)."
    assert image.shape[1:] == label.shape, "Image and label spatial dimensions must match (H, W, D)."

    input_depth = image.shape[-1]  # Original depth (D)

    # Compute scaling factor for depth dimension
    scale = output_depth / input_depth

    # Resize image (apply scaling along depth only)
    resized_image = np.stack([
        zoom(image[channel], (1, 1, scale), order=image_interpolation_order)
        for channel in range(image.shape[0])
    ], axis=0)

    # Resize label (apply scaling along depth only)
    resized_label = zoom(label, (1, 1, scale), order=label_interpolation_order)

    return resized_image, resized_label


def image_direction(array, direction):
    direction_matrix = np.array(direction).reshape(3, 3)

    reorder_axes = np.argsort(np.abs(direction_matrix).sum(axis=1))
    flip_axes = [i for i, row in enumerate(direction_matrix) if row[reorder_axes[i]] < 0]

    adjusted_array = np.transpose(array, reorder_axes[::-1])
    for axis in flip_axes:
        adjusted_array = np.flip(adjusted_array, axis)

    return adjusted_array.copy()


########################################################################################################################
def mode_filter(img, ksize=5):
    unfolded = F.unfold(img, kernel_size=ksize, padding=ksize//2)
    mode_values, _ = unfolded.long().mode(dim=1)
    filtered = mode_values.view(img.shape[2], img.shape[3])
    return filtered

def mode_filter_processing(img):
    img = img.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device='cuda')
    for i in range(img.shape[3]):
        img[:, :, i] = mode_filter(img[:, :, i])
    for i in range(img.shape[3]):
        img[:, :, :, i] = mode_filter(img[:, :, :, i])
    for i in range(img.shape[3]):
        img[:, :, :, :, i] = mode_filter(img[:, :, :, :, i])
    return img.squeeze(0).squeeze(0)

def get_proto_index(matrix, label_name, slicer, proto_num, threshold):

    index = matrix[label_name]
    ones_indices = np.where(index == 1)[0]
    
    distances = np.abs(ones_indices - slicer)
    alp = 1 - abs((ones_indices - slicer) / len(index))
    
    valid_indices = ones_indices[alp > threshold]
    valid_distances = distances[alp > threshold]
    valid_alp = alp[alp > threshold]
    
    sorted_indices = np.argsort(valid_distances)[:proto_num]
    
    proto_index = valid_indices[sorted_indices]
    filtered_alp = valid_alp[sorted_indices]

    return proto_index.tolist(), filtered_alp.tolist()



########################################################################################################################
def save_model(model, G_list, record_matrix, _config):
    save_path = _config["save_model_path"]
    os.makedirs(save_path, exist_ok=True)

    model_path = os.path.join(save_path, 'llfnet.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved at: {model_path}')
    
    proto_path = os.path.join(save_path, "proto_set.pkl")
    with open(proto_path, "wb") as f:
        pickle.dump(G_list, f)
    print(f'Proto list saved at: {proto_path}')
    
    matrix_path = os.path.join(save_path, 'record_matrix.csv')
    np.savetxt(matrix_path, record_matrix, delimiter=',')
    print(f'Record matrix saved at: {matrix_path}')

def saving(preds, filename, _config, direction=None, spacing=None, origin=None):
    base_path = _config.get("save_path", "./output")
    save_mhd = _config.get("save_mhd", True)
    save_nii = _config.get("save_nii", True)

    base_path_mhd = os.path.join(base_path, "mhd")
    base_path_nii = os.path.join(base_path, "nii")

    if save_mhd:
        os.makedirs(base_path_mhd, exist_ok=True)
    if save_nii:
        os.makedirs(base_path_nii, exist_ok=True)

    filepath_mhd = os.path.join(base_path_mhd, filename + ".mhd")
    filepath_nii = os.path.join(base_path_nii, filename + ".nii")

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    if preds.ndim == 4:
        preds = preds[0]

    direction = [float(x) for x in direction] if direction else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    spacing = [float(x) for x in spacing] if spacing else [1.0, 1.0, 1.0]
    origin = [float(x) for x in origin] if origin else [0.0, 0.0, 0.0]

    img = sitk.GetImageFromArray(preds)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)

    if save_mhd:
        sitk.WriteImage(img, filepath_mhd, useCompression=True)
        print(f"Saved compressed: {filepath_mhd}")

    if save_nii:
        sitk.WriteImage(img, filepath_nii)
        print(f"Saved uncompressed: {filepath_nii}")