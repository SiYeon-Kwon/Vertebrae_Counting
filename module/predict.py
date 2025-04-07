import torch
import torch.nn as nn
import nibabel as nib  # 用于读取医学 3D 图像格式（如 NIfTI）
import numpy as np
import os
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from dataset import MedicalDataset
from data_loader import load_nii_files
from model_forpredict import model_dict
import matplotlib.pyplot as plt

test_image_array, test_mask_array = load_nii_files("test")
test_dataset = MedicalDataset(test_image_array, test_mask_array)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 选择模型名称
model_name = 'unet3d'  # 可以修改为 'unet3dplus' 来选择不同的模型

# 根据模型名称获取模型类并实例化
ModelClass = model_dict[model_name]
model = ModelClass(in_channels=1, out_channels=1)

model.load_state_dict(torch.load('/home/eli/module/log'))
model.eval()  # 모델을 평가 모드로 전환

# 손실 함수 정의 (예시: binary cross entropy loss 사용)
criterion = torch.nn.BCEWithLogitsLoss()

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# IoU 계산 함수
def calculate_iou(pred_mask, true_mask, threshold=0.5):
    pred_mask_bin = (pred_mask > threshold).float()  # Threshold 적용하여 이진화
    intersection = torch.sum(pred_mask_bin * true_mask)
    union = torch.sum(pred_mask_bin) + torch.sum(true_mask) - intersection
    iou = intersection / union
    return iou.item()

# 정확도 계산 함수
def calculate_accuracy(pred_mask, true_mask, threshold=0.5):
    pred_mask_bin = (pred_mask > threshold).float()  # Threshold 적용하여 이진화
    correct = torch.sum(pred_mask_bin == true_mask)
    total = torch.numel(true_mask)
    accuracy = correct / total
    return accuracy.item()

def plot_3d_slices(volume, n_slices=8, axis=0, title="Volume Slices"):
    """
    volume: 3D numpy array
    n_slices: Number of slices to display
    axis: Axis along which to slice (0: Axial, 1: Coronal, 2: Sagittal)
    """
    # 슬라이스 축을 기준으로 3D 볼륨 슬라이스
    slices = np.linspace(0, volume.shape[axis] - 1, n_slices).astype(int)

    plt.figure(figsize=(20, 8))
    for i, slice_idx in enumerate(slices):
        plt.subplot(1, n_slices, i + 1)
        if axis == 0:  # Axial
            plt.imshow(volume[slice_idx, :, :], cmap='gray')
        elif axis == 1:  # Coronal
            plt.imshow(volume[:, slice_idx, :], cmap='gray')
        elif axis == 2:  # Sagittal
            plt.imshow(volume[:, :, slice_idx], cmap='gray')

        plt.axis('off')
        plt.title(f'Slice {slice_idx}')
    plt.suptitle(title)
    plt.show()

# 테스트 데이터셋에 대한 예측 수행 및 IoU, Loss, Accuracy 계산
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

total_loss = 0
total_iou = 0
total_accuracy = 0

with torch.no_grad():  # 예측 중에는 gradient 계산을 하지 않음
    for idx, (image, true_mask) in enumerate(test_loader):
        image = image.to(device)
        true_mask = true_mask.to(device)

        # 예측 마스크 계산
        pred_mask = model(image)
        pred_mask = torch.sigmoid(pred_mask)  # 출력 값을 [0, 1]로 스케일링

        # 손실 계산
        loss = criterion(pred_mask, true_mask)
        total_loss += loss.item()

        # IoU 계산
        iou = calculate_iou(pred_mask, true_mask)
        total_iou += iou

        # 정확도 계산
        accuracy = calculate_accuracy(pred_mask, true_mask)
        total_accuracy += accuracy

        # 결과 시각화 (첫 번째 테스트 샘플만 예시로 출력)
        if idx == 0:
            pred_mask_np = pred_mask.cpu().numpy().squeeze()
            true_mask_np = true_mask.cpu().numpy().squeeze()

            pred_mask_np = pred_mask.cpu().numpy().squeeze()  # 3D numpy array로 변환
            true_mask_np = true_mask.cpu().numpy().squeeze()  # 3D numpy array로 변환

            # 예측 마스크 3D 슬라이스 시각화
            plot_3d_slices(pred_mask_np, n_slices=8, axis=0, title="Predicted Mask - Axial Slices")
            plot_3d_slices(pred_mask_np, n_slices=8, axis=1, title="Predicted Mask - Coronal Slices")
            plot_3d_slices(pred_mask_np, n_slices=8, axis=2, title="Predicted Mask - Sagittal Slices")

            # 실제 마스크 3D 슬라이스 시각화
            plot_3d_slices(true_mask_np, n_slices=8, axis=0, title="True Mask - Axial Slices")
            plot_3d_slices(true_mask_np, n_slices=8, axis=1, title="True Mask - Coronal Slices")
            plot_3d_slices(true_mask_np, n_slices=8, axis=2, title="True Mask - Sagittal Slices")

            # 결과 시각화
            # slice_idx = pred_mask_np.shape[0] // 2  # 가운데 슬라이스 선택
            #
            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 2, 1)
            # plt.title("Predicted Mask")
            # plt.imshow(pred_mask_np[slice_idx], cmap='gray')
            #
            # plt.subplot(1, 2, 2)
            # plt.title("True Mask")
            # plt.imshow(true_mask_np[slice_idx], cmap='gray')
            #
            # plt.show()

# 평균 Loss, IoU, Accuracy 출력
num_samples = len(test_loader)
average_loss = total_loss / num_samples
average_iou = total_iou / num_samples
average_accuracy = total_accuracy / num_samples

print(f"Test Loss: {average_loss}")
print(f"Test IoU: {average_iou}")
print(f"Test Accuracy: {average_accuracy}")