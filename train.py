
# 导入必要的库和模块
import torch
import torch.nn as nn
from torch import optim
import timeit
from tqdm import tqdm
from utils import get_loaders
from model import Vit

# 设置设备（优先使用GPU，否则使用CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)  # 输出所使用的设备信息

# 超参数设置
EPOCHS = 50  # 训练的总轮数

BATCH_SIZE = 16  # 每批数据的大小

# 数据集路径
TRAIN_DF_DIR = "./dataset/train.csv"
TEST_DF_DIR = "./dataset/test.csv"
SUBMISSION_DF_DIR = "./dataset/sample_submission.csv"

# 模型参数
IN_CHANNELS = 1  # 输入图像的通道数
IMG_SIZE = 28  # 图像的尺寸
PATCH_SIZE = 4  # 分割图像的补丁大小
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS  # 嵌入维度，基于补丁大小和通道数计算
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 图像分割后的补丁数量
DROPOUT = 0.001  # Dropout层的丢弃概率

NUM_HEADS = 8  # 多头注意力机制的头数
ACTIVATION = "gelu"  # 激活函数类型
NUM_ENCODERS = 12  # 编码器的数量
NUM_CLASSES = 10  # 分类任务的类别数

LEARNING_RATE = 1e-4  # 学习率
ADAM_WEIGHT_DECAY = 0  # Adam优化器的权重衰减
ADAM_BETAS = (0.9, 0.999)  # Adam优化器的一阶和二阶矩估计的指数衰减率

# 获取数据加载器
train_dataloader, val_dataloader, test_dataloader = get_loaders(TRAIN_DF_DIR, TEST_DF_DIR, SUBMISSION_DF_DIR, BATCH_SIZE)

# 初始化模型
# Vit类实例化，传入模型参数
model = Vit(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT,
            NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASSES).to(device)

# 定义损失函数和优化器
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 使用Adam优化器，传入模型参数、学习率、权重衰减和一阶二阶矩估计的指数衰减率
optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)

# 开始计时
start = timeit.default_timer()

# 主训练循环
for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    model.train()  # 设置模型为训练模式

    # 初始化训练标签、预测结果列表和运行损失
    train_labels = []
    train_preds = []
    train_running_loss = 0

    # 遍历训练数据集
    for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        # 将图像和标签加载到指定设备上
        img = img_label["image"].float().to(device)
        label = img_label["label"].type(torch.uint8).to(device)

        # 前向传播
        y_pred = model(img)
        # 获取预测标签
        y_pred_label = torch.argmax(y_pred, dim=1)

        # 更新训练标签和预测结果
        train_labels.extend(label.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())

        # 计算损失
        loss = criterion(y_pred, label)

        # 清零梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # 累加损失
        train_running_loss += loss.item()

    # 计算训练集平均损失
    train_loss = train_running_loss / (idx + 1)

    # 评估模型
    model.eval()  # 设置模型为评估模式

    # 初始化验证标签、预测结果列表和运行损失
    val_labels = []
    val_preds = []
    val_running_loss = 0

    # 关闭梯度计算
    with torch.no_grad():
        # 遍历验证数据集
        for idx, img_label in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            # 将图像和标签加载到指定设备上
            img = img_label["image"].float().to(device)
            label = img_label["label"].type(torch.uint8).to(device)

            # 前向传播
            y_pred = model(img)
            # 获取预测标签
            y_pred_label = torch.argmax(y_pred, dim=1)

            # 更新验证标签和预测结果
            val_labels.extend(label.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())

            # 计算损失
            loss = criterion(y_pred, label)
            # 累加损失
            val_running_loss += loss.item()

    # 计算验证集平均损失
    val_loss = val_running_loss / (idx + 1)

    # 打印训练和验证的结果
    print("-" * 30)
    print(f"Train Loss Epoch {epoch+1} : {train_loss:.4f}")
    print(f"Val Loss Epoch {epoch+1} : {val_loss:.4f}")

    # 计算并打印训练准确率
    train_accuracy = sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels)
    print(f"Train Accuracy EPOCH {epoch + 1}: {train_accuracy:.4f}")

    # 计算并打印验证准确率
    val_accuracy = sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels)
    print(f"Val Accuracy EPOCH {epoch + 1}: {val_accuracy:.4f}")
    print("-" * 30)

# 结束计时并打印训练总耗时
stop = timeit.default_timer()
print(f"Training Time:{stop - start:.2f}s")