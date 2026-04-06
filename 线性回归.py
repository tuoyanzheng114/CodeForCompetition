import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# debug中文乱码
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']  # macOS/Windows 通用
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== 1. 读取文件（使用实际路径） ==========
file_path = '农村收支.csv'
df = pd.read_csv(file_path)

print("数据前5行：")
print(df.head())

# ========== 2. 定义变量 ==========
X = df[['Disposable_Income']].values   # 自变量：可支配收入（二维数组）
y = df['Consumption_Expenditure'].values  # 因变量：消费支出

print(f"\n样本数：{len(X)}")

# ========== 3. 划分训练集和测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 4. 训练线性回归模型 ==========
model = LinearRegression()
model.fit(X_train, y_train)

# 模型参数
slope = model.coef_[0]      # 斜率 = 边际消费倾向
intercept = model.intercept_
print(f"\n边际消费倾向 (MPC): {slope:.4f}")
print(f"截距 (自主性消费): {intercept:.2f}")

# ========== 5. 评估模型 ==========
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差 (MSE): {mse:.2f}")
print(f"决定系数 (R²): {r2:.4f}")

# ========== 6. 可视化（全部数据 + 回归线） ==========
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.7, label='实际数据')
# 绘制回归线
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = model.predict(X_range)
plt.plot(X_range, y_range, color='red', linewidth=2, label='回归线')
plt.xlabel('可支配收入 (元)')
plt.ylabel('消费支出 (元)')
plt.title('消费支出与可支配收入的线性关系')
plt.legend()
plt.grid(True)
plt.show()

# ========== 7. 打印预测示例 ==========
print("\n测试集预测示例（真实值 vs 预测值）：")
for i in range(min(5, len(y_test))):
    print(f"  可支配收入: {X_test[i][0]:8.2f} -> 真实消费: {y_test[i]:8.2f}, 预测消费: {y_pred[i]:8.2f}")