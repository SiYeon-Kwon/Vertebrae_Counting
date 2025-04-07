import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib

'''class MedicalDataset(Dataset):
    """
    自定义医学数据集，用于按需加载和预处理医学图像和掩码。
    """
    def __init__(self, image_paths, mask_paths):
        """
        初始化数据集。

        参数：
        - image_paths (list): 图像文件路径列表
        - mask_paths (list): 掩码文件路径列表
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        """
        返回数据集的大小。

        返回：
        - 数据集的长度（int）
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。

        参数：
        - idx (int): 索引

        返回：
        - image (Tensor): 预处理后的图像张量
        - mask (Tensor): 预处理后的掩码张量
        """
        # 按需从磁盘加载图像和掩码
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # 将numpy数组转换为张量，并添加通道维度
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, Depth, Height, Width)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)    # (1, Depth, Height, Width)

        # 使用三线性插值将图像和掩码调整为统一的尺寸 (64, 64, 64)
        image = F.interpolate(image.unsqueeze(0), size=(64, 64, 64), mode='trilinear', align_corners=True).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(64, 64, 64), mode='nearest').squeeze(0)
        # 注意：对于掩码，使用 'nearest' 插值避免引入新类别

        return image, mask

if __name__ == "__main__":
    # 示例：创建数据集并查看形状
    from data_loader import load_nii_files

    # 加载文件路径
    train_image_paths, train_mask_paths = load_nii_files("train")

    train_dataset = MedicalDataset(train_image_paths, train_mask_paths)
    print(f"数据集大小：{len(train_dataset)}")

    # 取出一个样本查看形状
    image, mask = train_dataset[0]
    print(f"图像形状：{image.shape}")
    print(f"掩码形状：{mask.shape}")'''

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import torchio as tio  # Import the TorchIO library for medical image augmentations

class MedicalDataset(Dataset):
    """
    自定义医学数据集，用于按需加载和预处理医学图像和掩码，并支持数据增强。
    """
    def __init__(self, image_paths, mask_paths, augment=False):
        """
        初始化数据集。

        参数：
        - image_paths (list): 图像文件路径列表
        - mask_paths (list): 掩码文件路径列表
        - augment (bool): 是否应用数据增强
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

        # 定义数据增强变换
        self.transform = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2)),                  # 随机翻转
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10), # 随机旋转和缩放
            tio.RandomElasticDeformation(max_displacement=5),# 随机弹性形变
            tio.RandomNoise(mean=0, std=0.1)                 # 随机噪声
        ])

    def __len__(self):
        """
        返回数据集的大小。

        返回：
        - 数据集的长度（int）
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。

        参数：
        - idx (int): 索引

        返回：
        - image (Tensor): 预处理后的图像张量
        - mask (Tensor): 预处理后的掩码张量
        """
        # 加载图像和掩码
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # 转换为张量并添加通道维度
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # 统一尺寸 (64, 64, 64)
        image = F.interpolate(image.unsqueeze(0), size=(64, 64, 64), mode='trilinear', align_corners=True).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(64, 64, 64), mode='nearest').squeeze(0)

        # 如果启用数据增强，对图像和掩码进行变换
        if self.augment:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask)
            )
            transformed = self.transform(subject)
            image = transformed['image'].data
            mask = transformed['mask'].data

        return image, mask
