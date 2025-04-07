import torch

def iou(preds, targets, smooth=1e-6):
    """
    计算交并比（IoU）。

    参数：
    - preds (Tensor): 模型预测输出
    - targets (Tensor): 实际标签
    - smooth (float): 平滑项，防止除以零

    返回：
    - float: 平均IoU值
    """
    preds = preds > 0.5  # 阈值化，得到二值预测
    targets = targets > 0.5  # 二值化实际标签

    intersection = (preds & targets).float().sum(dim=(1, 2, 3, 4))  # 计算交集
    union = (preds | targets).float().sum(dim=(1, 2, 3, 4))        # 计算并集

    iou_score = (intersection + smooth) / (union + smooth)  # 计算IoU
    return iou_score.mean()  # 返回平均IoU

def dice_loss(preds, targets, smooth=1e-6):
    """
    计算Dice损失。

    参数：
    - preds (Tensor): 模型预测输出
    - targets (Tensor): 实际标签
    - smooth (float): 平滑项，防止除以零

    返回：
    - float: Dice损失值
    """
    preds = preds.sigmoid()  # 应用sigmoid函数
    targets = targets > 0.5  # 二值化实际标签

    intersection = (preds * targets).sum(dim=(1, 2, 3, 4))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(1, 2, 3, 4)) + targets.sum(dim=(1, 2, 3, 4)) + smooth)

    return 1 - dice.mean()  # 返回Dice损失

if __name__ == "__main__":
    # 示例：计算IoU和Dice损失
    preds = torch.randn(1, 1, 64, 64, 64)
    targets = torch.randint(0, 2, (1, 1, 64, 64, 64))

    iou_score = iou(preds, targets)
    dice = dice_loss(preds, targets)

    print(f"IoU得分：{iou_score}")
    print(f"Dice损失：{dice}")
