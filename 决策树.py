import re
import warnings

import matplotlib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 禁用部分警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*precision is ill-defined.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*recall is ill-defined.*")

# 尽量使用常见字体，避免中文字体缺失导致脚本报大量警告
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_text_column(text):
    """从 text 字段中提取数值特征。"""
    keys = ["收入", "工资", "经营", "财产", "转移", "消费",
            "食品", "衣着", "居住", "交通", "教育", "医疗"]
    pattern = r"(" + "|".join(keys) + r")\s*(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, str(text))
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
    返回：过滤后的 DataFrame 和被删除的行索引
    """
    if feature_columns is None:
        feature_columns = df.columns.tolist()

    mask = pd.Series([True] * len(df), index=df.index)
    for col in feature_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        mask = mask & col_mask
        print(f"特征 {col}: 下限={lower_bound:.2f}, 上限={upper_bound:.2f}, 保留 {col_mask.sum()} / {len(df)}")

    removed_idx = df.index[~mask]
    df_clean = df.loc[mask].reset_index(drop=True)
    return df_clean, removed_idx


def split_datasets(csv_path="农村收支.csv"):
    """读取数据、解析特征、去异常值，并按 6:2:2 划分数据集。"""
    df_raw = pd.read_csv(csv_path)

    parsed = df_raw["text"].apply(parse_text_column)
    feature_df = pd.DataFrame(parsed.tolist())

    if "city" in feature_df.columns:
        feature_df = feature_df.drop("city", axis=1)

    # 只针对“收入”列去除异常值，避免样本删除过多
    feature_df_clean, removed_indices = remove_outliers_iqr(
        feature_df,
        feature_columns=["收入"],
        multiplier=1.5
    )

    target_clean = df_raw["target"].drop(index=removed_indices).reset_index(drop=True)

    # 缺失值用中位数填补，适合数值型特征
    imputer = SimpleImputer(strategy="median")
    feature_df_clean = pd.DataFrame(
        imputer.fit_transform(feature_df_clean),
        columns=feature_df_clean.columns
    )

    le = LabelEncoder()
    target_encoded = le.fit_transform(target_clean)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        feature_df_clean,
        target_encoded,
        test_size=0.2,
        random_state=42,
        stratify=target_encoded
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val
    )

    print(f"\n原始样本数: {len(df_raw)}")
    print(f"删除异常值后样本数: {len(feature_df_clean)}")
    print(f"删除的行索引（原始 CSV 中的行号）: {removed_indices.tolist()}")
    print("\n划分结果：")
    print("训练集：", x_train.shape, y_train.shape)
    print("验证集：", x_val.shape, y_val.shape)
    print("测试集：", x_test.shape, y_test.shape)

    return x_train, x_val, x_test, y_train, y_val, y_test, le


def print_metrics(name, y_true, y_pred, label_encoder):
    """打印模型评估结果。"""
    print(f"\n{name}性能：")
    print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
    print("分类报告：")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )


def tune_decision_tree(x_train, y_train, x_val, y_val):
    """使用验证集选择更合适的决策树参数。"""
    candidate_params = [
        {"criterion": "gini", "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "gini", "max_depth": 4, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "gini", "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "entropy", "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "entropy", "max_depth": 4, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "entropy", "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "entropy", "max_depth": 6, "min_samples_split": 4, "min_samples_leaf": 2},
        {"criterion": "gini", "max_depth": None, "min_samples_split": 4, "min_samples_leaf": 2},
    ]

    best_model = None
    best_score = -1
    best_params = None

    for params in candidate_params:
        model = DecisionTreeClassifier(random_state=42, **params)
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_val_pred)
        print(f"参数 {params} -> 验证集准确率: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    print("\n最优参数：", best_params)
    print(f"最优验证集准确率: {best_score:.4f}")
    return best_model


def train_and_evaluate_decision_tree(x_train, x_val, x_test, y_train, y_val, y_test, label_encoder):
    """训练、选参、评估并可视化决策树。"""
    clf = tune_decision_tree(x_train, y_train, x_val, y_val)

    y_train_pred = clf.predict(x_train)
    y_val_pred = clf.predict(x_val)
    y_test_pred = clf.predict(x_test)

    print_metrics("训练集", y_train, y_train_pred, label_encoder)
    print_metrics("验证集", y_val, y_val_pred, label_encoder)
    print_metrics("测试集", y_test, y_test_pred, label_encoder)

    cm = confusion_matrix(y_test, y_test_pred)
    print("测试集混淆矩阵：")
    print(cm)

    feature_name_map = {
        "收入": "income",
        "工资": "salary",
        "经营": "business",
        "财产": "property",
        "转移": "transfer",
        "消费": "expense",
        "食品": "food",
        "衣着": "clothing",
        "居住": "housing",
        "交通": "transport",
        "教育": "education",
        "医疗": "medical",
    }
    class_name_map = {"低": "low", "中": "middle", "高": "high"}
    plot_feature_names = [feature_name_map.get(col, col) for col in x_train.columns]
    plot_class_names = [class_name_map.get(name, name) for name in label_encoder.classes_]

    plt.figure(figsize=(16, 10))
    plot_tree(
        clf,
        filled=True,
        rounded=True,
        fontsize=9,
        feature_names=plot_feature_names,
        class_names=plot_class_names
    )
    plt.title("Decision Tree of Rural Income and Expenditure")
    plt.tight_layout()
    plt.savefig("决策树可视化.png", dpi=300)
    plt.close()
    print("决策树图已保存为: 决策树可视化.png")

    return clf


if __name__ == "__main__":
    x_train, x_val, x_test, y_train, y_val, y_test, label_encoder = split_datasets("农村收支.csv")
    model = train_and_evaluate_decision_tree(
        x_train, x_val, x_test, y_train, y_val, y_test, label_encoder
    )