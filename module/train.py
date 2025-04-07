import torch
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from dataset import MedicalDataset
from model import model_dict
from metrics import dice_loss, iou
from data_loader import load_nii_files
import os


def train_one_epoch(train_loader, model, optimizer, device):
    model.train()  # 设置模型为训练模式
    total_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        # 清空缓存，释放GPU内存
        torch.cuda.empty_cache()

        outputs = model(images)
        loss = dice_loss(outputs, masks)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


def validate_one_epoch(val_loader, model, device):
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    total_iou = 0.0
    valid_samples_count = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            if images is None or masks is None:
                continue
            if images.size(0) == 0 or masks.size(0) == 0:
                continue
            torch.cuda.empty_cache()

            outputs = model(images)
            loss = dice_loss(outputs, masks)
            total_loss += loss.item()

            iou_score = iou(outputs, masks)
            total_iou += iou_score.item()
            valid_samples_count += 1
            avg_loss = total_loss / valid_samples_count
            avg_iou  = total_iou /  valid_samples_count if valid_samples_count > 0 else 0.0

    return avg_loss,avg_iou


if __name__ == '__main__':
    # 加载训练和验证数据的文件路径
    train_image_paths, train_mask_paths = load_nii_files("train")
    val_image_paths, val_mask_paths = load_nii_files("val")

    # 创建数据集和数据加载器
    train_dataset = MedicalDataset(train_image_paths, train_mask_paths)
    val_dataset = MedicalDataset(val_image_paths, val_mask_paths)

    # 设置 batch_size 和 num_workers
    batch_size = 2  # 可以尝试设置为更小的值
    num_workers = 16  # 根据您的CPU核心数进行调整

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 选择模型名称
    model_name = 'unet3d'  # 可以修改为 'unet3dplus' 来选择不同的模型

    # 根据模型名称获取模型类并实例化
    ModelClass = model_dict[model_name]
    model = ModelClass(in_channels=1, out_channels=1)

    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    checkpoint_path = f'{model_name}_best_epoch.pth'
    best_val_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if os.path.exists(checkpoint_path):
        print(f"加载已有的模型检查点：{checkpoint_path}")

        model.load_state_dict(torch.load(checkpoint_path))
        #checkpoint = torch.load(checkpoint_path)
        #model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型状态字典
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态字典
        #best_val_loss = checkpoint['best_val_loss']  # 加载最佳验证损失
        #print(f"继续训练，之前最佳验证损失为: {best_val_loss:.4f}")

    log_file_path = f'{model_name}_training_log.txt'
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:
            log_file.write("Epoch, Train Loss, Val Loss, Val IOU\n")



    # 开始训练循环
    num_epochs = 500

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(train_loader, model, optimizer, device)
        val_loss, val_iou = validate_one_epoch(val_loader, model, device)

        print(
            f'第 {epoch + 1}/{num_epochs} 轮，训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}')

        # 保存验证损失最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{model_name}_best_epoch.pth')
            print(f"在第 {epoch + 1} 轮保存模型，验证损失: {val_loss:.4f}")

        if (epoch + 1) % 50 == 0:
            checkpoint_path_epoch = f'{model_name}_epoch_{epoch + 1}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path_epoch)
            print(f'保存第 {epoch + 1} 轮的模型')

            # 将每一轮的损失和 IOU 写入日志文件
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}, {val_iou:.4f}\n")