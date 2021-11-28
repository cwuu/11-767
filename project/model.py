import torch
import torch.nn as nn
import torch.nn.functional as F

class depthwise_pointwise_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super(depthwise_pointwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, groups=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Lighter_SeeInDark(nn.Module):
    def __init__(self, num_classes=10):
        super(Lighter_SeeInDark, self).__init__()
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.conv1_1 = depthwise_pointwise_conv(4, 32, kernel_size=3, padding=1)
        self.conv1_2 = depthwise_pointwise_conv(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = depthwise_pointwise_conv(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = depthwise_pointwise_conv(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = depthwise_pointwise_conv(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = depthwise_pointwise_conv(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = depthwise_pointwise_conv(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = depthwise_pointwise_conv(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = depthwise_pointwise_conv(256, 512, kernel_size=3, padding=1)
        self.conv5_2 = depthwise_pointwise_conv(512, 512, kernel_size=3, padding=1)
        
        # up 
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = depthwise_pointwise_conv(512, 256, kernel_size=3, padding=1)
        self.conv6_2 = depthwise_pointwise_conv(256, 256, kernel_size=3, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = depthwise_pointwise_conv(256, 128, kernel_size=3, padding=1)
        self.conv7_2 = depthwise_pointwise_conv(128, 128, kernel_size=3, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = depthwise_pointwise_conv(128, 64, kernel_size=3, padding=1)
        self.conv8_2 = depthwise_pointwise_conv(64, 64, kernel_size=3, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = depthwise_pointwise_conv(64, 32, kernel_size=3, padding=1)
        self.conv9_2 = depthwise_pointwise_conv(32, 32, kernel_size=3, padding=1)
        
        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        
        conv10= self.conv10_1(conv9)
        out = nn.functional.pixel_shuffle(conv10, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    #def lrelu(self, x):
    #    outt = torch.max(0.2*x, x)
    #    return outt

class Light_SeeInDark_Rewrite(nn.Module):
    def __init__(self, num_classes=10):
        super(Light_SeeInDark_Rewrite, self).__init__()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        #self.conv1_2 = depthwise_pointwise_conv(32, 32, kernel_size=3, padding=1)
        self.conv1_2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32)
        self.conv1_2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, groups=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.conv2_2 = depthwise_pointwise_conv(64, 64, kernel_size=3, padding=1)
        self.conv2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.conv2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, groups=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        #self.conv3_2 = depthwise_pointwise_conv(128, 128, kernel_size=3, padding=1)
        self.conv3_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128)
        self.conv3_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, groups=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        #self.conv4_2 = depthwise_pointwise_conv(256, 256, kernel_size=3, padding=1)
        self.conv4_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256)
        self.conv4_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, groups=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        #self.conv5_2 = depthwise_pointwise_conv(512, 512, kernel_size=3, padding=1)
        self.conv5_2_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, groups=512)
        self.conv5_2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, groups=1)
        
        # up 
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        #self.conv6_2 = depthwise_pointwise_conv(256, 256, kernel_size=3, padding=1)
        self.conv6_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256)
        self.conv6_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, groups=1)
        
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        #self.conv7_2 = depthwise_pointwise_conv(128, 128, kernel_size=3, padding=1)
        self.conv7_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128)
        self.conv7_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, groups=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        #self.conv8_2 = depthwise_pointwise_conv(64, 64, kernel_size=3, padding=1)
        self.conv8_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.conv8_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, groups=1)
        
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        #self.conv9_2 = depthwise_pointwise_conv(32, 32, kernel_size=3, padding=1)
        self.conv9_2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32)
        self.conv9_2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, groups=1)
        
        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        #conv1 = self.lrelu(self.conv1_2(conv1))
        conv1 = self.conv1_2_1(conv1)
        conv1 = self.lrelu(self.conv1_2_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        #conv2 = self.lrelu(self.conv2_2(conv2))
        conv2 = self.conv2_2_1(conv2)
        conv2 = self.lrelu(self.conv2_2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        #conv3 = self.lrelu(self.conv3_2(conv3))
        conv3 = self.conv3_2_1(conv3)
        conv3 = self.lrelu(self.conv3_2_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        #conv4 = self.lrelu(self.conv4_2(conv4))
        conv4 = self.conv4_2_1(conv4)
        conv4 = self.lrelu(self.conv4_2_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.lrelu(self.conv5_1(pool4))
        #conv5 = self.lrelu(self.conv5_2(conv5))
        conv5 = self.conv5_2_1(conv5)
        conv5 = self.lrelu(self.conv5_2_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        #conv6 = self.lrelu(self.conv6_2(conv6))
        conv6 = self.conv6_2_1(conv6)
        conv6 = self.lrelu(self.conv6_2_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        #conv7 = self.lrelu(self.conv7_2(conv7))
        conv7 = self.conv7_2_1(conv7)
        conv7 = self.lrelu(self.conv7_2_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        #conv8 = self.lrelu(self.conv8_2(conv8))
        conv8 = self.conv8_2_1(conv8)
        conv8 = self.lrelu(self.conv8_2_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        #conv9 = self.lrelu(self.conv9_2(conv9))
        conv9 = self.conv9_2_1(conv9)
        conv9 = self.lrelu(self.conv9_2_2(conv9))
        
        conv10= self.conv10_1(conv9)
        out = nn.functional.pixel_shuffle(conv10, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    #def lrelu(self, x):
    #    outt = torch.max(0.2*x, x)
    #    return outt


class Light_SeeInDark(nn.Module):
    def __init__(self, num_classes=10):
        super(Light_SeeInDark, self).__init__()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = depthwise_pointwise_conv(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = depthwise_pointwise_conv(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = depthwise_pointwise_conv(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = depthwise_pointwise_conv(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = depthwise_pointwise_conv(512, 512, kernel_size=3, padding=1)
        
        # up 
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = depthwise_pointwise_conv(256, 256, kernel_size=3, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = depthwise_pointwise_conv(128, 128, kernel_size=3, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = depthwise_pointwise_conv(64, 64, kernel_size=3, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = depthwise_pointwise_conv(32, 32, kernel_size=3, padding=1)
        
        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        
        conv10= self.conv10_1(conv9)
        out = nn.functional.pixel_shuffle(conv10, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    #def lrelu(self, x):
    #    outt = torch.max(0.2*x, x)
    #    return outt


class Quantized_Light_SeeInDark(nn.Module):
    def __init__(self, model_fp32):
        super(Quantized_Light_SeeInDark, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

class SeeInDark(nn.Module):
    def __init__(self, num_classes=10):
        super(SeeInDark, self).__init__()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        # up 
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        
        conv10= self.conv10_1(conv9)
        out = nn.functional.pixel_shuffle(conv10, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    #def lrelu(self, x):
    #    outt = torch.max(0.2*x, x)
    #    return outt