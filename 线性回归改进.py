import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ================== 1. 特征解析函数（与原代码一致） ==================
def parse_text_column(text):
    """从字符串中提取数值特征"""
    keys = ['收入', '工资', '经营', '财产', '转移', '消费',
            '食品', '衣着', '居住', '交通', '教育', '医疗']
    pattern = r'(' + '|'.join(keys) + r')\s*(-?\d+(?:\.\d+)?)'
    matches = re.findall(pattern, text)
    result = {key: None for key in keys}
    for key, value in matches:
        result[key] = float(value)
    return result

def remove_outliers_iqr(df, feature_columns=None, multiplier=1.5):
    """基于 IQR 删除异常值"""
    if feature_columns is None:
        feature_columns = df.columns.tolist()
    mask = pd.Series([True] * len(df))
    for col in feature_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        mask = mask & col_mask
        print(f"特征 {col}: 下限={lower_bound:.2f}, 上限={upper_bound:.2f}, 保留 {col_mask.sum()} / {len(df)}")
    removed_idx = df.index[~mask]
    df_clean = df[mask].reset_index(drop=True)
    return df_clean, removed_idx

# ================== 2. 准备回归数据（统一划分流程） ==================
def prepare_regression_data(csv_path="农村收支.csv"):
    """读取数据、解析特征、去除异常值、划分训练/验证/测试集（6:2:2）"""
    # 读取原始数据
    df_raw = pd.read_csv(csv_path)

    # 解析 text 列，得到特征 DataFrame
    parsed = df_raw['text'].apply(parse_text_column)
    feature_df = pd.DataFrame(parsed.tolist())

    # 提取自变量 X（收入）和因变量 y（消费）
    X_raw = feature_df[['收入']].copy()   # 保持 DataFrame 格式，便于后续处理
    y_raw = feature_df['消费'].copy()

    # 去除异常值（仅针对“收入”特征，与原代码一致）
    X_clean, removed_indices = remove_outliers_iqr(X_raw, feature_columns=['收入'], multiplier=1.5)
    # 同步删除 y 中的对应行
    y_clean = y_raw.drop(index=removed_indices).reset_index(drop=True)
    X_clean = X_clean.reset_index(drop=True)

    print(f"\n原始样本数: {len(df_raw)}")
    print(f"删除异常值后样本数: {len(X_clean)}")
    print(f"删除的行索引（原始CSV中的行号）: {removed_indices.tolist()}")

    # 划分训练集、验证集、测试集 (6:2:2)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
    )

    print("\n划分结果：")
    print("训练集：", X_train.shape, y_train.shape)
    print("验证集：", X_val.shape, y_val.shape)
    print("测试集：", X_test.shape, y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

# ================== 3. 线性回归模型训练与评估 ==================
if __name__ == "__main__":
    # 获取统一划分的数据
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_regression_data("农村收支.csv")

    # 训练线性回归模型（使用训练集）
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 模型参数
    slope = model.coef_[0]      # 边际消费倾向 MPC
    intercept = model.intercept_
    print(f"\n边际消费倾向 (MPC): {slope:.4f}")
    print(f"截距 (自主性消费): {intercept:.2f}")

    # 在测试集上评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"测试集均方误差 (MSE): {mse:.2f}")
    print(f"测试集决定系数 (R²): {r2:.4f}")

    #在验证集上评估（用于调参，此处仅展示）
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    print(f"验证集均方误差 (MSE): {val_mse:.2f}")
    print(f"验证集决定系数 (R²): {val_r2:.4f}")

    # 可视化（使用全部清理后的数据）
    # 合并 X_clean 和 y_clean 用于绘图
    X_clean_full = pd.concat([X_train, X_val, X_test], axis=0)
    y_clean_full = pd.concat([y_train, y_val, y_test], axis=0)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_clean_full, y_clean_full, alpha=0.7, label='实际数据')
    # 绘制回归线
    X_range = np.linspace(X_clean_full.min(), X_clean_full.max(), 100).reshape(-1, 1)
    y_range = model.predict(X_range)
    plt.plot(X_range, y_range, color='red', linewidth=2, label='回归线')
    plt.xlabel('收入 (元)')
    plt.ylabel('消费支出 (元)')
    plt.title('消费支出与收入的线性关系')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 打印测试集预测示例
    print("\n测试集预测示例（真实值 vs 预测值）：")
    for i in range(min(5, len(y_test))):
        print(f"  收入: {X_test.iloc[i, 0]:8.2f} -> 真实消费: {y_test.iloc[i]:8.2f}, 预测消费: {y_pred[i]:8.2f}")