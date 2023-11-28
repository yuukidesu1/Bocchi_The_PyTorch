import torch.nn as nn                               # 这个模块包含了构建神经网络模型所需要的各种类和函数
from torch.hub import load_state_dict_from_url      # 这个函数通常用于从预训练的模型仓库中加载模型的权重参数


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):         # 调用父类的构造函数来初始化这个自定义的神经网络模型
        super(VGG, self).__init__()
        self.features = features                            # 定义模型的特征提取部分
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))         # 定义自适应平均池化层，将输入的特征图尺寸调整为(7, 7)
        # 定义分类器部分，包括几个全连接层和非线性激活函数
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    '''
    这段代码定义了VGG模型前向传播函数 'forward'。这个函数接受一个输出张量x，
    按照一定的顺序对输入进行处理并返回一系列特征图
    '''
    def forward(self, x):
        # 使用注释掉的代码进行前向传播（逐层处理）
        # x = self.features(x)                          # 特征提取部分
        # x = self.avgpool(x)                           # 平均池化
        # x = torch.flatten(x, 1)                       # 将特征图展平为一维
        # x = self.classifier(x)                        # 分类器部分

        # 使用切片操作对输入进行逐层处理，分别获取不同层的特征图
        feat1 = self.features[:4](x)                    # 前4个卷积层的特征
        feat2 = self.features[4:9](feat1)               # 5个
        feat3 = self.features[9:16](feat2)              # 7个
        feat4 = self.features[16:23](feat3)             # 7个
        feat5 = self.features[23:-1](feat4)             # 最后一层卷积层之前的特征

        # 返回特征图列表
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        # 遍历模型的所有子模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 如果子模块是卷积层，使用凯明初始化方法初始化权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 如果卷积层有偏置项，初始化为常数0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 如果子模块是批量归一化层，初始化权重和偏置项为常数
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 如果子模块是线性层，初始化权重为正态分布（均值0，方差1），偏置项为0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            # 如果配置中的值是‘M’，则添加一个最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 如果配置中的值不是‘M’，则添加一个卷积层
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # 如果启用了批量归一化，添加卷积层、批量归一化层和ReLU激活函数
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                # 如果没有启用批量归一化，添加卷积层和ReLU激活函数
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels=3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=False, in_channels=in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)

    del model.avgpool
    del model.classifier
    return model

vgg_model = VGG16(pretrained=True)

features = vgg_model.features

for i, layer in enumerate(features):
    if isinstance(layer, nn.Conv2d):
        print(f"Convolutional layer {i+1} output channels:{layer.out_channels}")