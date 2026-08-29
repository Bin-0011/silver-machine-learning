# ==================== 1. 导入必须的库 ====================
import torch
import torch.nn as nn

# ==================== 2. 定义神经网络类 ====================
class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ---------- 第一组：提取简单特征 ----------
        # 输入 [B, 3, 32, 32] -> 输出 [B, 8, 32, 32]
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 8,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )
        # ReLU 激活函数
        # 不是 Relu ,也不是 ReLu
        self.relu1 = nn.ReLU()

        # 池化 Pooling 层  下采样
        # [B, 8, 32, 32] -> [B, 8, 16, 16]
        self.pooling1 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        # ---------- 第二组：提取高级特征 ----------
        # 卷积特征提取
        # 输入 [B, 8, 16, 16] -> 输出 [B, 16, 16, 16]
        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )
        # ReLu 激活函数
        self.relu2 = nn.ReLU()
        # 池化 Pooling 层
        # [B, 16, 16, 16] -> [B, 16, 8, 8]
        self.pooling2 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )
        
        # ---------- 分类头 ----------
        # 输入 [B, 1024] -> 输出 [B, 10] (Logits)
        # 非隐藏层可近似将 out_features 看作 num_classes
        # 全连接层 Linear(in_features, out_features) 设置为 (1024, 10)
        # self.fc 为什么写成 fc 而不是 linear 或 dense 或 fully_connected 或 mlp 或 classifier 呢？ 为什么不是 logits？
        self.fc = nn.Linear(
            in_features = 16 * 8 * 8,
            out_features = 10
        )

    # ==================== 3. 定义前向传播（数据流动逻辑） ====================
    def forward(self, x):
        # 输入 x: [B, 3, 32, 32]
        print("提取简单特征:")
        # 调用的时候是 x = self.conv1(x) 而不是 x = conv1(x) 因为 conv1 是类的属性，必须通过 self 来访问
        # x = x.conv1() 这样写是错误的，因为 x 是一个张量对象，它没有 conv1 方法
        x = self.conv1(x)
        print("x:", x)
        print("x.shape:", x.shape)
        x = self.relu1(x)
        print("x.shape:", x.shape)
        x = self.pooling1(x)
        print("x.shape:", x.shape)

        print("提取高级特征:")
        x = self.conv2(x)
        print("x:", x)
        print("x.shape:", x.shape)
        x = self.relu2(x)
        print("x.shape:", x.shape)
        x = self.pooling2(x)
        print("x.shape:", x.shape)

        # 展开：把多维张量展平成一维张量，方便后续的全连接层处理
        print("展开:")
        # 可以使用 x = x.view(x.size(0), -1) 代替
        x = torch.flatten(x, start_dim=1)
        print("x:", x)
        print("x.shape:", x.shape)

        # 全连接分类
        print("分类头:")
        logits = self.fc(x)
        print("logits:", logits)
        print("logits.shape:", logits.shape)

        return logits

# ==================== 4. 测试代码（验证 Shape） ====================
if __name__ == "__main__":
    # 创建虚拟输入 (模拟 1 张 32x32 的彩色图片)
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # 实例化模型
    model = MiniCNN()
    
    # 前向传播
    output = model(dummy_input)
    
    # 打印最终结果
    print(f"输入图片 Shape: {dummy_input.shape}")
    print(f"最终输出 Logits Shape: {output.shape}")