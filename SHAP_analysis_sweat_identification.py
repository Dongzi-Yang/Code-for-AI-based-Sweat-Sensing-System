import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap
import seaborn as sns
import plotly.graph_objects as go  # 用于绘制 Sankey 图

# 1. 导入数据
file_path = r'D:\工作\工作\课题相关文件\课题2-数据\APP数据\20240909\特征提取\Overall_Feature.xlsx'
res = pd.read_excel(file_path)

# 2. 划分训练集和测试集
temp = np.random.permutation(len(res))  # 随机排列索引

# 使用实际数据长度来划分训练集和测试集
train_size = int(0.8 * len(res))  # 训练集大小为数据集80%
P_train = res.iloc[temp[:train_size], :80].values
T_train = res.iloc[temp[:train_size], 80].values
P_test = res.iloc[temp[train_size:], :80].values  # 剩余部分作为测试集
T_test = res.iloc[temp[train_size:], 80].values

# 3. 数据归一化
scaler = MinMaxScaler()
P_train = scaler.fit_transform(P_train)
P_test = scaler.transform(P_test)

# 4. 数据平铺为适合CNN的格式
P_train = P_train.reshape((len(P_train), 1, 8, 10))  # 每个样本包含8个传感器，每个传感器10个特征
P_test = P_test.reshape((len(P_test), 1, 8, 10))

# 5. 目标数据编码
T_train_encoded = T_train - 1  # 将类别从 1-6 转换为 0-5 的索引
T_test_encoded = T_test - 1

# 将数据转换为PyTorch的Dataset和DataLoader
train_dataset = TensorDataset(torch.tensor(P_train, dtype=torch.float32),
                              torch.tensor(T_train_encoded, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(P_test, dtype=torch.float32), torch.tensor(T_test_encoded, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 6. 构造CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)

        # 计算展平后的大小
        self._to_linear = None
        self._initialize_weights()

        self.fc1 = nn.Linear(self._to_linear, 4)  # 修改为6个输出

    def _initialize_weights(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 8, 10)
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # 动态展平
        x = torch.softmax(self.fc1(x), dim=1)
        return x


# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 训练模型
num_epochs = 500
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 8. 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {100 * correct / total}%')

# 9. 保存模型
torch.save(model.state_dict(), 'my_CNN_model_3.pth')
print("Model saved successfully.")

# 10. 加载模型并进行 SHAP 分析
model.load_state_dict(torch.load('my_CNN_model_3.pth'))
model.eval()

# 准备所有测试数据
test_images, test_labels = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))

# 创建 SHAP 解释器
explainer = shap.GradientExplainer(model, test_images)

# 计算 SHAP 值
shap_values = explainer.shap_values(test_images)

# 转换为 NumPy 数组
shap_values = np.array(shap_values)
test_images_np = test_images.numpy()[:, 0, :, :]  # 从 (num_samples, 1, 8, 10) 变为 (num_samples, 8, 10)

# 输出shap_values和test_images的形状进行调试
print("shap_values shape:", shap_values.shape)
print("test_images shape:", test_images_np.shape)

# 对每个传感器的10个特征的SHAP值求和，保留类别维度
shap_sum_per_sensor = np.sum(shap_values, axis=3)  # 对每个传感器的10个特征求和，形状为 (120, 1, 8, 6)

# 选取某个类或所有类的平均值
# 例如，取所有类别的平均值
shap_mean_per_sensor = np.mean(shap_sum_per_sensor, axis=-1).reshape(len(test_images_np), -1)  # 形状变为 (120, 8)

# 对所有样本取平均，得到每个传感器的总体贡献
shap_mean_per_sensor_total = np.mean(shap_mean_per_sensor, axis=0)

# 准备传感器的名称
sensor_names = [f'Sensor {i+1}' for i in range(8)]

# 绘制每个传感器的 SHAP 值
plt.figure(figsize=(8, 6))
plt.bar(sensor_names, shap_mean_per_sensor_total, color='blue', alpha=0.7)
plt.xlabel('Sensor')
plt.ylabel('Mean SHAP Value')
plt.title('Mean SHAP Value per Sensor')
plt.show()

# ---- Sankey 图 ----
# 定义传感器和分类标签
sensors = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']
classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4']

# 创建源和目标节点的索引
source_indices = []
target_indices = []
values = []
colors = []  # 存储颜色，根据 SHAP 值的正负性区分

# 传感器（源）到类别（目标）的 SHAP 值流
# 遍历每个传感器，并获取每个传感器的总 SHAP 值
for sensor_idx in range(len(shap_mean_per_sensor_total)):
    shap_value = shap_mean_per_sensor_total[sensor_idx]  # 获取某个传感器的 SHAP 值
    print(f"Sensor {sensor_idx + 1} SHAP value: {shap_value}")




# 定义 Sankey 图的节点和链接
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=sensors + classes,  # 节点标签：传感器 + 类别
        color="blue"  # 节点颜色（可以自定义）
    ),
    link=dict(
        source=source_indices,  # 源节点索引
        target=target_indices,  # 目标节点索引
        value=values,  # 链接的值（SHAP 值的绝对值）
        color=colors  # 根据 SHAP 值的正负性设定颜色
    )
))

# 设置图的标题并显示图表
fig.update_layout(title_text="Sankey Diagram of SHAP Values: Sensor Contribution to Classifications", font_size=10)
fig.show()

# 显示混淆矩阵
predictions = []
actuals = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

cm = confusion_matrix(actuals, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(T_test))
disp.plot()
plt.title('Confusion Matrix for Test Data')
plt.show()

# 可视化传感器对每个类别分类结果的影响
categories = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
plt.figure(figsize=(12, 8))
for i in range(4):
    plt.bar(range(1, 9), shap_mean_per_sensor_per_class[:, i], alpha=0.6, label=categories[i])


plt.xlabel('Sensor Index')
plt.ylabel('Mean SHAP Value')
plt.title('Sensor Impact on Classification Result for Each Class')
plt.legend(loc='upper right')
plt.show()