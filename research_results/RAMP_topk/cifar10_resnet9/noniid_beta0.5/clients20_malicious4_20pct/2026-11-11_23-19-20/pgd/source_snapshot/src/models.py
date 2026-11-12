from resnet import ResNet9
from vgg import VGG
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST dataset
    Architecture: Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> FC -> FC
    """
    def __init__(self, num_classes=10, args=None):
        super(SimpleCNN, self).__init__()
        self.args = args  # Store args for compatibility with other components
        # First convolutional layer: 1 input channel, 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After two pooling layers: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_resnet18_cifar10(num_classes=10, args=None):
    """
    ResNet-18 for CIFAR-10 (32x32 images)
    Modified from torchvision ResNet-18 for smaller input size
    """
    model = models.resnet18(pretrained=False)
    
    # Modify first conv layer for CIFAR-10 (32x32 instead of 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    
    # Modify final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def get_mobilenetv2_cifar10(num_classes=10, args=None):
    """
    MobileNetV2 for CIFAR-10 (32x32 images)
    Lightweight model with depthwise separable convolutions
    """
    model = models.mobilenet_v2(pretrained=False)
    
    # Modify first conv layer for CIFAR-10
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Modify classifier for CIFAR-10
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    return model


def get_model(data, args):
    # 支持通过 --model 参数选择模型（默认使用原有的 ResNet9）
    model_name = getattr(args, 'model', 'resnet9')
    
    if data == 'cifar10':
        if model_name == 'resnet9':
            model = ResNet9(3, num_classes=10, args=args)
        elif model_name == 'resnet18':
            model = get_resnet18_cifar10(num_classes=10, args=args)
        elif model_name == 'vgg9':
            model = VGG('VGG9', num_classes=10)
        elif model_name == 'mobilenetv2':
            model = get_mobilenetv2_cifar10(num_classes=10, args=args)
        else:
            # 默认使用 ResNet9（向后兼容）
            model = ResNet9(3, num_classes=10, args=args)
    elif data == 'cifar100':
        # 使用 ResNet9 替代 VGG9，因为：
        # 1. ResNet9 有残差连接，训练更稳定
        # 2. 参数量更少（6.6M vs 11.6M），通信开销小
        # 3. 在联邦学习的 Non-IID 场景下更鲁棒
        # 4. BatchNorm 配置更适合联邦学习（track_running_stats=False）
        model = ResNet9(3, num_classes=100, args=args)
    elif data == 'tinyimagenet':
        model = get_resnet18_64x64()
    elif data == 'mnist':
        # MNIST uses simple CNN (28x28 grayscale images)
        model = SimpleCNN(num_classes=10, args=args)
    elif data == 'fmnist':
        # Fashion-MNIST uses ResNet9 with 1 channel
        model = ResNet9(1, num_classes=10, args=args)

    return model
         

def get_resnet18_64x64(num_classes=200):
    """
    Returns a ResNet-18 model modified for 64x64 input images and a specific number of output classes.

    Args:
        num_classes (int): The number of output classes. Default is 200.

    Returns:
        model (torch.nn.Module): Modified ResNet-18 model.
    """
    # Load the ResNet-18 model pretrained on ImageNet
    model = models.resnet18(pretrained=False)

    # Modify the first convolutional layer to handle 64x64 images
    model.conv1 = nn.Conv2d(
        in_channels=3, 
        out_channels=64, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    
    # Adjust the max pooling layer to fit smaller image sizes
    model.maxpool = nn.Identity()

    # Modify the fully connected layer to output num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model