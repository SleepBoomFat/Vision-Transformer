
import torch
import torch.nn as nn

# 带有patch嵌入、位置嵌入和分类token的Transformer前缀模块
class PatchEmbedding(nn.Module):
    """
    Patch嵌入模块，用于将输入图像转换为patch特征向量。

    参数:
    - in_channels: 输入图像的通道数。
    - patch_size: patch的大小。
    - embed_dim: patch嵌入后的维度。
    - num_patches: 图像划分后的patch数量。
    - dropout: 堆叠中的dropout比例。
    """
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        # 定义一个卷积层，用于将输入图像分割为patch，并将其嵌入到指定维度
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        # 定义分类token，用于模型学习全局的分类表示
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        # 定义位置嵌入，用于为每个patch添加位置信息
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 扩展分类token，使其与输入批次大小相同
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 通过patcher将输入图像转换为patch嵌入，并调整维度顺序
        x = self.patcher(x).permute(0, 2, 1)
        # 将分类token与patch嵌入拼接在一起
        x = torch.cat([cls_token, x], dim=1)
        # 添加位置嵌入
        x = x + self.position_embedding
        # 应用dropout
        x = self.dropout(x)
        return x

# Vision Transformer (ViT) 主模型
class Vit(nn.Module):
    """
    Vision Transformer (ViT) 模型。

    参数:
    - in_channels: 输入图像的通道数。
    - patch_size: patch的大小。
    - embed_dim: patch嵌入后的维度。
    - num_patches: 图像划分后的patch数量。
    - dropout: 堆叠中的dropout比例。
    - num_heads: 注意力头的数量。
    - activation: 激活函数。
    - num_encoders: Transformer编码器层数。
    - num_classes: 分类任务的类别数。
    """
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout,
                 num_heads, activation, num_encoders, num_classes):
        super(Vit, self).__init__()
        # 初始化patch嵌入模块
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)
        # 初始化Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                   activation=activation,
                                                   batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        # 初始化MLP层，用于最终的分类预测
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        # 通过patch嵌入模块转换输入图像
        x = self.patch_embedding(x)
        # 通过Transformer编码器块处理嵌入后的图像
        x = self.encoder_blocks(x)
        # 从输出中提取第一个元素（即带有全局信息的分类token），进行分类预测
        x = self.MLP(x[:, 0, :])
        return x

if __name__ == "__main__":
    # 配置模型参数
    IMG_SIZE = 224
    IN_CHANNELS = 3
    PATCH_SIZE = 16
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 49
    EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS  # 16
    DROPOUT = 0.001
    NUM_HEADS = 8
    ACTIVATION = "gelu"
    NUM_ENCODERS = 4
    NUM_CLASSES = 10
    HIDDEN_LAYER = 768

    # 根据可用设备选择运行模式
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 实例化ViT模型，并转移到指定设备
    # model = PatchEmbedding(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT)
    model = Vit(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT, NUM_HEADS, ACTIVATION, NUM_ENCODERS,
                NUM_CLASSES).to(device)
    # 生成随机输入数据
    x = torch.randn(size=(1, 3, 224, 224)).to(device)
    # 前向传播，获取预测结果
    prediction = model(x)
    # 打印输出形状，预期为(batch_size, num_classes)
    print(prediction.shape)