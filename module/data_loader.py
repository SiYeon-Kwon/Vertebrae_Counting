import os

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
    从指定的数据类型（train、val、test）中获取图像和掩码文件的路径列表。

    参数：
    - data_type (str): 数据集类型，"train"、"val" 或 "test"

    返回：
    - image_paths (list): 图像文件路径的列表
    - mask_paths (list): 掩码文件路径的列表
    """
    image_paths = []
    mask_paths = []

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

        # 如果同时存在图像和掩码文件，则保存路径
        if image_path and mask_path:
            print(f"找到图像：{image_path}")
            print(f"找到掩码：{mask_path}")

            image_paths.append(image_path)
            mask_paths.append(mask_path)
        else:
            print(f"警告：患者 {patient_number} 缺少图像或掩码文件")

    return image_paths, mask_paths


if __name__ == "__main__":
    # 加载训练、验证和测试数据的文件路径
    train_image_paths, train_mask_paths = load_nii_files("train")
    val_image_paths, val_mask_paths = load_nii_files("val")
    test_image_paths, test_mask_paths = load_nii_files("test")

    # 打印加载的文件路径列表（可选）
    print(train_image_paths)
    print(val_image_paths)
    print(test_image_paths)
