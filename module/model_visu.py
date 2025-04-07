import torch
#from thop import clever_format, profile
from torchsummary import summary

from model import UNet3D





if __name__ == "__main__":
    input_shape = [64, 64]
    #num_classes = 3
    #backbone = 'vgg'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(1, 1).to(device)  # bins = [1, 2, 18, 26],
    summary(model, (1,1,64, 64,64))

    dummy_input = torch.randn(1,1,64, 64,64).to(device)
    #flops, params = profile(model.to(device), (dummy_input,), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    # --------------------------------------------------------#
   # flops = flops * 2
    #flops, params = clever_format([flops, params], "%.3f")
   # print('Total GFLOPS: %s' % (flops))
    #print('Total params: %s' % (params))
