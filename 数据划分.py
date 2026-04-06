import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    """
    基于 IQR 删除异常值。
    df: 特征 DataFrame
    feature_columns: 要检查的列名列表，若为 None 则检查所有数值列
    multiplier: IQR 的倍数，默认 1.5
    返回：过滤后的 DataFrame 和 被删除的行索引
    """
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
    # 1. 读取原始数据
    df_raw = pd.read_csv(csv_path)
    
    # 2. 解析 text 列，得到特征 DataFrame
    parsed = df_raw['text'].apply(parse_text_column)
    feature_df = pd.DataFrame(parsed.tolist())
    # 去掉城市列（如果有）
    if 'city' in feature_df.columns:
        feature_df = feature_df.drop('city', axis=1)
    
    # 3.去除异常值 ———— 去掉最高和最低
    
    #想针对“收入”一个特征去掉极端值，可以这样
    feature_df_clean, removed_indices = remove_outliers_iqr(feature_df, feature_columns=['收入'], multiplier=1.5)
    #这里提供对所有特征值去掉极端值的版本，但是可能会删除过多样本，导致数据量不足，所以默认只针对“收入”特征去掉极端值
    # feature_df_clean, removed_indices = remove_outliers_iqr(feature_df, multiplier=1.5)
    
    # 同步删除目标列中的对应行
    target_clean = df_raw['target'].drop(index=removed_indices).reset_index(drop=True)
    
    print(f"\n原始样本数: {len(df_raw)}")
    print(f"删除异常值后样本数: {len(feature_df_clean)}")
    print(f"删除的行索引（原始CSV中的行号）: {removed_indices.tolist()}")
    
    # 4. 编码目标变量（低/中/高 -> 0/1/2）
    le = LabelEncoder()
    target_encoded = le.fit_transform(target_clean)
    
    # 5. 划分训练集、验证集、测试集（6:2:2）
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        feature_df_clean, target_encoded, test_size=0.2, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    print("\n划分结果：")
    print("训练集：", x_train.shape, y_train.shape)
    print("验证集：", x_val.shape, y_val.shape)
    print("测试集：", x_test.shape, y_test.shape)
    
    return (x_train, x_val, x_test, y_train, y_val, y_test, le)

if __name__ == "__main__":
    split_datasets()