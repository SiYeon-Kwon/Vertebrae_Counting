import torch
import torch.nn as nn
from module.Aspp import ASPP, attention_block
from module.attention import se_block,cbam_block,eca_block,CA_Block
from module.inception import InceptionModule

attention_block = [se_block, cbam_block, eca_block, CA_Block]

class UNet3D(nn.Module):
    """
    3D U-Net 模型，用于医学图像分割。
    """
    def __init__(self, in_channels, out_channels,phi = 2):
        super(UNet3D, self).__init__()
        self.phi = phi
        # 编码器部分
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # 瓶颈层
        self.bottleneck = self.conv_block(512, 1024)

        # 解码器部分
        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)

        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        if 1 <= self.phi and self.phi <= 4:
            self.feat1_att = attention_block[self.phi - 1](64)
            self.feat2_att = attention_block[self.phi - 1](128)
            self.feat3_att = attention_block[self.phi - 1](256)
            self.feat4_att = attention_block[self.phi - 1](512)

        self.aspp = ASPP(512, 512,1)
        self.inception = InceptionModule(in_channels=512,out_1x1=128,red_3x3=96,out_3x3=192,red_5x5=32,out_5x5=64,out_pool=128)

        # 最后的1x1卷积层
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        卷积块，包括两次卷积和激活函数。
        """
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """
        上采样块，使用转置卷积实现上采样。
        """
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        """
        前向传播。
        """
        # 编码路径
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool3d(kernel_size=2,stride=1,padding=1)(enc1))
        enc3 = self.encoder3(nn.MaxPool3d(kernel_size=2,stride=1,padding=1)(enc2))
        enc4 = self.encoder4(nn.MaxPool3d(kernel_size=2,stride=1,padding=1)(enc3))

        # 瓶颈层
        bottleneck = self.aspp(enc4)
        bottleneck = self.inception(bottleneck)
        bottleneck = self.bottleneck(nn.MaxPool3d(2)(bottleneck))


        if 1<= self.phi and self.phi <= 4:
            enc1 = self.feat1_att(enc1)
            enc2 = self.feat2_att(enc2)
            enc3 = self.feat3_att(enc3)
            enc4 = self.feat4_att(enc4)

        # 解码路径，带有跳跃连接

        #print(f"enc4 shape: {enc4.shape}")
        dec4 = self.upconv4(bottleneck)
        #print(f"dec4 shape: {dec4.shape}")
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # 最终输出
        return self.final_conv(dec1)


class UNet3Dplus(nn.Module):
    """
    3D U-Net++ 模型，用于医学图像分割。
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化模型。

        参数：
        - in_channels (int): 输入通道数
        - out_channels (int): 输出通道数
        """
        super(UNet3Dplus, self).__init__()
        # 编码器部分
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # 瓶颈层
        self.bottleneck = self.conv_block(512, 1024)

        # 解码器部分
        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)

        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # 最后的1x1卷积层
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        卷积块，包括两次卷积和激活函数。

        参数：
        - in_channels (int): 输入通道数
        - out_channels (int): 输出通道数

        返回：
        - nn.Sequential: 卷积块
        """
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """
        上采样块，使用转置卷积实现上采样。

        参数：
        - in_channels (int): 输入通道数
        - out_channels (int): 输出通道数

        返回：
        - nn.ConvTranspose3d: 转置卷积层
        """
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        """
        前向传播。

        参数：
        - x (Tensor): 输入张量

        返回：
        - Tensor: 输出张量
        """
        # 编码路径
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool3d(kernel_size=2,stride=1,padding=1)(enc1))
        enc3 = self.encoder3(nn.MaxPool3d(kernel_size=2,stride=1,padding=1)(enc2))
        enc4 = self.encoder4(nn.MaxPool3d(kernel_size=2,stride=1,padding=1)(enc3))

        # 瓶颈层
        bottleneck = self.bottleneck(nn.MaxPool3d(2)(enc4))

        # 解码路径，带有跳跃连接
        dec4 = self.upconv4(bottleneck)
        dec4 = self.crop_and_concat(dec4, enc4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.crop_and_concat(dec3, enc3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.crop_and_concat(dec2, enc2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.crop_and_concat(dec1, enc1)
        dec1 = self.decoder1(dec1)

        # 最终输出
        return self.final_conv(dec1)

    def crop_and_concat(self, upconv_output, encoder_output):
        """
        裁剪编码器输出并与上采样结果拼接。

        参数：
        - upconv_output (Tensor): 上采样输出
        - encoder_output (Tensor): 编码器输出

        返回：
        - Tensor: 拼接后的张量
        """
        # 计算尺寸差异
        diff_z = encoder_output.size(2) - upconv_output.size(2)
        diff_y = encoder_output.size(3) - upconv_output.size(3)
        diff_x = encoder_output.size(4) - upconv_output.size(4)

        # 裁剪编码器输出
        encoder_output_cropped = encoder_output[:, :,
            diff_z // 2: encoder_output.size(2) - diff_z // 2,
            diff_y // 2: encoder_output.size(3) - diff_y // 2,
            diff_x // 2: encoder_output.size(4) - diff_x // 2]

        # 在通道维度上拼接
        return torch.cat((upconv_output, encoder_output_cropped), dim=1)


model_dict = {
    'unet3d': UNet3D,
    'unet3dplus': UNet3Dplus
}


if __name__ == "__main__":
    model_name = 'unet3d'  # 或 'unet3dplus'
    ModelClass = model_dict[model_name]
    model = ModelClass(in_channels=1, out_channels=1)
    input_tensor = torch.randn(1, 1, 64, 64, 64)  # (batch_size, channels, depth, height, width)
    output = model(input_tensor)
    print(f"输出形状：{output.shape}")