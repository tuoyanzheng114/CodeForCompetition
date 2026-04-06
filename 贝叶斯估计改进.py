import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置中文字体（防止乱码）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 数据解析与划分函数 ----------
def parse_text_column(text):
    keys = ['收入', '工资', '经营', '财产', '转移', '消费', 
            '食品', '衣着', '居住', '交通', '教育', '医疗']
    pattern = r'(' + '|'.join(keys) + r')\s*(-?\d+(?:\.\d+)?)'
    matches = re.findall(pattern, text)
    result = {key: None for key in keys}
    for key, value in matches:
        result[key] = float(value)
    return result

def remove_outliers_iqr(df, feature_columns=None, multiplier=1.5):
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

def split_datasets(csv_path="农村收支.csv"):
    df_raw = pd.read_csv(csv_path)
    parsed = df_raw['text'].apply(parse_text_column)
    feature_df = pd.DataFrame(parsed.tolist())
    if 'city' in feature_df.columns:
        feature_df = feature_df.drop('city', axis=1)
    # 只对“收入”做 IQR 异常值剔除（可调整 multiplier）
    feature_df_clean, removed_indices = remove_outliers_iqr(feature_df, feature_columns=['收入'], multiplier=1.5)
    target_clean = df_raw['target'].drop(index=removed_indices).reset_index(drop=True)
    
    le = LabelEncoder()
    target_encoded = le.fit_transform(target_clean)
    
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        feature_df_clean, target_encoded, test_size=0.2, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.25, random_state=42
    )
    return x_train, x_val, x_test, y_train, y_val, y_test, le

# ---------- 贝叶斯分类器（图形 + 终端输出） ----------
def bayes_classifier(csv_path="农村收支.csv"):
    # 1. 获取划分好的数据集
    x_train, x_val, x_test, y_train, y_val, y_test, label_encoder = split_datasets(csv_path)
    
    # 2. 训练高斯朴素贝叶斯模型
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    
    # 3. 验证集准确率
    y_val_pred = gnb.predict(x_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\n验证集准确率: {val_acc:.4f}")
    
    # 4. 测试集准确率
    y_test_pred = gnb.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 5. 分类报告（文本）
    all_labels = list(range(len(label_encoder.classes_)))
    print("\n分类报告（测试集）：")
    print(classification_report(y_test, y_test_pred, 
                                labels=all_labels, 
                                target_names=label_encoder.classes_,
                                zero_division=0))
    
    # 6. 混淆矩阵（并绘制图形）
    cm = confusion_matrix(y_test, y_test_pred, labels=all_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('朴素贝叶斯分类混淆矩阵（测试集）')
    plt.show()
    
    # 7. 打印预测示例
    print("\n测试集前5个样本的真实 vs 预测：")
    for i in range(min(5, len(x_test))):
        true_label = label_encoder.inverse_transform([y_test[i]])[0]
        pred_label = label_encoder.inverse_transform([y_test_pred[i]])[0]
        print(f"样本{i+1}: 真实={true_label}, 预测={pred_label}")
    
    return gnb, label_encoder

if __name__ == "__main__":
    model, le = bayes_classifier("农村收支.csv")