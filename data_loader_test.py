import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

# 设置基础路径和各数据集的子文件夹名称
base_path = "/workspace/eli"

folders = {
    "train": "01_training",
    "val": "02_validation",
    "test": "03_test"
}

# 定义可能的图像和掩码文件名模式
possible_image_patterns = [
    "{patient_number}_CT-ax.nii.gz",
    "{patient_number}.nii.gz",
    "{patient_number}_CT-iso.nii.gz"
]

possible_mask_patterns = [
    "{patient_number}_CT-ax_seg.nii.gz",
    "{patient_number}_seg.nii.gz",
    "{patient_number}_CT-iso_seg.nii.gz"
]


def load_nii_files(data_type):
    """
    从指定的数据类型（train、val、test）中加载NIfTI格式的图像和掩码文件。

    参数：
    - data_type (str): 数据集类型，"train"、"val" 或 "test"

    返回：
    - image_arrays (list): 图像数据的列表
    - mask_arrays (list): 掩码数据的列表
    """
    image_arrays = []
    mask_arrays = []

    # 构建数据集的路径
    folder_path = os.path.join(base_path, folders[data_type])

    # 获取患者编号列表
    patients_numbers = os.listdir(folder_path)

    for patient_number in patients_numbers:
        image_path = None
        mask_path = None

        # 查找图像文件路径
        for img_pattern in possible_image_patterns:
            temp_image_path = os.path.join(
                folder_path, patient_number, img_pattern.format(patient_number=patient_number))
            if os.path.exists(temp_image_path):
                image_path = temp_image_path
                break

        # 查找掩码文件路径
        for mask_pattern in possible_mask_patterns:
            temp_mask_path = os.path.join(
                folder_path, patient_number, mask_pattern.format(patient_number=patient_number))
            if os.path.exists(temp_mask_path):
                mask_path = temp_mask_path
                break

        # 如果同时存在图像和掩码文件，则加载数据
        if image_path and mask_path:
            print(f"正在加载图像：{image_path}")
            print(f"正在加载掩码：{mask_path}")

            image = nib.load(image_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()

            image_arrays.append(image)
            mask_arrays.append(mask)
        else:
            print(f"警告：患者 {patient_number} 缺少图像或掩码文件")

    return image_arrays, mask_arrays


if __name__ == "__main__":
    # 获取训练集患者编号列表
    train_folder_path = os.path.join(base_path, folders["train"])
    all_patients_numbers = os.listdir(train_folder_path)
    first_10_patients = all_patients_numbers[:10]  # 只取前10个患者

    # 定义一个函数，用于替换 os.listdir，只返回前10个患者编号
    def limited_listdir(path):
        if path == train_folder_path:
            return first_10_patients
        else:
            return os.listdir(path)

    # 使用 patch 替换 os.listdir，只在加载训练集时生效
    with patch('os.listdir', side_effect=limited_listdir):
        # 加载训练集数据（只加载前10个患者）
        train_image_array, train_mask_array = load_nii_files("train")

    # 不加载验证集和测试集，或者根据需要加载
    # val_image_array, val_mask_array = load_nii_files("val")
    # test_image_array, test_mask_array = load_nii_files("test")

    # 可视化训练数据中的一个样本
    if train_image_array and train_mask_array:
        idx = len(train_image_array) // 2  # 使用中间的图像
        slice_idx = train_image_array[idx].shape[2] // 2  # 选择z轴中间的切片

        img = train_image_array[idx][:, :, slice_idx]
        mask = train_mask_array[idx][:, :, slice_idx]

        fig, arr = plt.subplots(1, 2, figsize=(14, 10))

        # 显示CT图像
        arr[0].imshow(img, cmap="gray")
        arr[0].set_title('CT 图像')
        arr[0].set_aspect('equal')  # 设置1:1比例

        # 显示分割掩码
        arr[1].imshow(mask, cmap="gray")
        arr[1].set_title('分割掩码')
        arr[1].set_aspect('equal')  # 设置1:1比例

        plt.show()
    else:
        print("没有可用的图像或掩码数据。")

    # 打印加载的数据列表（可选）
    print(train_image_array)
    # print(val_image_array)
    # print(test_image_array)
