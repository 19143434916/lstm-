import os
import numpy as np
import pandas as pd

import time as t   # ← 新增这一行

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from lstm_load_forecasting import lstm

# ================== 全局配置 ==================
# 修改为你的实际项目路径
BASE_DIR     = r"D:\Downloads\lstm-load-forecasting-master\lstm-load-forecasting-master"
DATA_PATH    = os.path.join(BASE_DIR, "data", "11minutes.csv")
MODEL_CAT_ID = "NF01"
MODEL_DIR    = os.path.join(BASE_DIR, f"{MODEL_CAT_ID}_models")
RES_DIR      = os.path.join(BASE_DIR, f"{MODEL_CAT_ID}_results")
TIMESTEPS    = 60        # 滑动窗口长度
BATCH_SIZE   = 64        # NF01 所有模型都是 64
TEST_RATIO   = 0.2       # 后 20% 做测试（前 80% 训练）

# 与训练时一致的特征 & 目标
# 与旧 NF01 模型保持一致：11 维特征
FEATURE_COLS = [
    "n_packets",
    "n_bytes",
    "n_dest_asn",
    "n_dest_ports",
    "n_dest_ip",
    "tcp_udp_ratio_packets",
    "tcp_udp_ratio_bytes",
    "dir_ratio_packets",
    "dir_ratio_bytes",
    "avg_duration",
    "avg_ttl",
]

TARGET_COL   = "n_flows"   # 原始列名，实际会拷贝到 actual


def _load_and_prepare_data():
    """读取 11minutes.csv，构造特征 + train/test 划分 + 标准化，返回一堆东西。"""

    df_raw = pd.read_csv(DATA_PATH)

    # 如果有 minute_index，就设成索引，方便和你现在前端对上
    if "minute_index" in df_raw.columns:
        df_raw = df_raw.set_index("minute_index")

    # 目标列统一叫 actual（和之前一致）
    df_raw["actual"] = df_raw[TARGET_COL].astype(float)

    # hour_of_day / day_index 如果文件里没有，就从 id_time 衍生
    if ("hour_of_day" not in df_raw.columns) or ("day_index" not in df_raw.columns):
        if "id_time" not in df_raw.columns:
            raise ValueError(
                "数据中没有 hour_of_day/day_index，也没有 id_time，"
                "没办法自动构造时间特征，请检查 11minutes.csv。"
            )
        ts = pd.to_datetime(df_raw["id_time"])
        # 小时可以用浮点数（小时 + 分钟/60），和训练时保持一致就好
        df_raw["hour_of_day"] = ts.dt.hour + ts.dt.minute / 60.0
        day0 = ts.dt.normalize().min()
        df_raw["day_index"] = (ts.dt.normalize() - day0).dt.days

    # 取出特征和标签
    X_full = df_raw[FEATURE_COLS].astype(float)
    y_full = df_raw["actual"].astype(float)

    n_total   = len(df_raw)
    split_idx = int(n_total * (1 - TEST_RATIO))

    X_train_raw, X_test_raw = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train_raw, y_test_raw = y_full.iloc[:split_idx], y_full.iloc[split_idx:]

    # ===== 标准化（和训练时一样：只用 train 拟合）=====
    scaler_X = StandardScaler()
    X_train = pd.DataFrame(
        scaler_X.fit_transform(X_train_raw),
        index=X_train_raw.index,
        columns=X_train_raw.columns,
    )
    X_test = pd.DataFrame(
        scaler_X.transform(X_test_raw),
        index=X_test_raw.index,
        columns=X_test_raw.columns,
    )

    scaler_y = StandardScaler()
    y_train = pd.Series(
        scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1)).reshape(-1),
        index=y_train_raw.index,
        name="actual",
    )
    y_test = pd.Series(
        scaler_y.transform(y_test_raw.values.reshape(-1, 1)).reshape(-1),
        index=y_test_raw.index,
        name="actual",
    )

    return (
        df_raw,
        X_train_raw,
        X_test_raw,
        y_train_raw,
        y_test_raw,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler_X,
        scaler_y,
    )


def run_model_and_build_df(
    model_name: str,
    save_csv: bool = True,
    csv_suffix: str | None = None,
) -> pd.DataFrame:
    """
    根据指定的 NF01 模型，在测试集上做一次真实预测，
    返回一个 DataFrame：
        index: minute_index（测试集后段）
        columns: ['y_true', 'y_pred', 'residual']

    参数
    ----
    model_name : str
        比如 'NF01_2_l-32_d-0.1'（不要带 .h5/.keras 后缀）
    save_csv : bool
        是否把结果另存为 CSV（放在 NF01_results 目录下）
    csv_suffix : str | None
        输出文件名可选后缀（默认用今天日期）。

    返回
    ----
    df_pred : pd.DataFrame
        y_true / y_pred / residual 的结果表。
    """

    # 1. 读数据 + 标准化
    (
        df_raw,
        X_train_raw,
        X_test_raw,
        y_train_raw,
        y_test_raw,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler_X,
        scaler_y,
    ) = _load_and_prepare_data()

    # 2. 加载模型（优先 .keras，没有再找 .h5）
    base = os.path.join(MODEL_DIR, model_name)
    model_path_keras = base + ".keras"
    model_path_h5    = base + ".h5"

    if os.path.exists(model_path_keras):
        model_path = model_path_keras
    elif os.path.exists(model_path_h5):
        model_path = model_path_h5
    else:
        raise FileNotFoundError(f"找不到模型文件：{model_path_keras} 或 {model_path_h5}")

    model = load_model(model_path)

    # 3. 在测试集上做预测（注意：这里传的是“已标准化”的 X_test）
    y_pred_scaled = lstm.get_predictions(
        model=model,
        X=X_test,
        batch_size=BATCH_SIZE,
        timesteps=TIMESTEPS,
        verbose=0,
    )
    y_pred_scaled = np.asarray(y_pred_scaled).reshape(-1)

    # 对应的 y_true（原始尺度）——要把前 TIMESTEPS 个丢掉，对齐滑动窗口
    # X_test 长度 = N_test，总预测点数 = N_test - TIMESTEPS
    y_true_full = y_test_raw.values  # 原始尺度
    y_true_aligned = y_true_full[TIMESTEPS:]
    index_aligned = y_test_raw.index[TIMESTEPS:]

    # 4. 把预测值从标准化尺度反变换回原始单位
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    # 5. 计算残差 & 组装 DataFrame
    residual = y_true_aligned - y_pred

    df_pred = pd.DataFrame(
        {
            "y_true": y_true_aligned,
            "y_pred": y_pred,
            "residual": residual,
        },
        index=index_aligned,
    )
    df_pred.index.name = "minute_index"

    # 6. 可选：保存 CSV，前端如果还想用静态文件，可以继续用这种格式
    if save_csv:
        os.makedirs(RES_DIR, exist_ok=True)
        if csv_suffix is None:
            csv_suffix = t.strftime("%Y%m%d")
        out_path = os.path.join(
            RES_DIR, f"{model_name}_pred_vs_actual_{csv_suffix}.csv"
        )
        df_pred.to_csv(out_path, sep=";")
        print("Prediction CSV saved to:", out_path)

    return df_pred


# ... [保持您原有的所有代码不变，并在文件末尾添加以下函数] ...

# 在你的 nf01_utils.py 文件末尾，替换 get_dashboard_data 函数为以下代码：

def get_dashboard_data(model_name: str, threshold_sigma: float = 2.5) -> pd.DataFrame:
    """
    运行模型并构建用于前端仪表板的 DataFrame，包含异常检测字段。
    """

    # 1. 调用核心预测函数
    df_pred_res = run_model_and_build_df(model_name=model_name, save_csv=False)

    if df_pred_res.empty:
        return pd.DataFrame()

    # 2. 计算残差的标准差 (Sigma)
    std_residual = df_pred_res["residual"].std()
    if std_residual < 1e-6:
        std_residual = 1.0

    # 3. 计算异常检测的绝对阈值
    # 调整：使用更宽松的阈值，或者使用百分位数
    threshold_value = threshold_sigma * std_residual
    
    # 【方案A】使用百分位数作为阈值（推荐）
    # 只有残差超过95%分位数的才算异常
    percentile_threshold = np.percentile(np.abs(df_pred_res["residual"]), 95)
    threshold_value = max(threshold_value, percentile_threshold)
    
    print(f"Sigma阈值: {threshold_sigma * std_residual:.2f}, 95%分位阈值: {percentile_threshold:.2f}")
    print(f"最终使用阈值: {threshold_value:.2f}")

    # 4. 执行异常检测
    df_pred_res["isAnomaly"] = np.abs(df_pred_res["residual"]) > threshold_value

    # 5. 计算上下界
    df_pred_res["upperBound"] = df_pred_res["y_pred"] + threshold_value
    df_pred_res["lowerBound"] = df_pred_res["y_pred"] - threshold_value

    # 6. 重置索引
    df_dashboard = df_pred_res.reset_index()
    
    # 7. 重命名列
    rename_map = {}
    if 'minute_index' in df_dashboard.columns:
        rename_map['minute_index'] = 'minute'
    elif 'index' in df_dashboard.columns:
        rename_map['index'] = 'minute'
    
    if 'y_true' in df_dashboard.columns:
        rename_map['y_true'] = 'actual'
    if 'y_pred' in df_dashboard.columns:
        rename_map['y_pred'] = 'predicted'
    
    df_dashboard = df_dashboard.rename(columns=rename_map)
    
    # 8. 确保有 minute 列
    if 'minute' not in df_dashboard.columns:
        df_dashboard['minute'] = range(len(df_dashboard))
    
    # 9. 选取需要的列
    required_cols = ["minute", "actual", "predicted", "residual", "upperBound", "lowerBound", "isAnomaly"]
    existing_cols = [col for col in required_cols if col in df_dashboard.columns]
    
    df_dashboard = df_dashboard[existing_cols].round(2)

    # 统计异常数量
    anomaly_count = df_dashboard["isAnomaly"].sum()
    total_count = len(df_dashboard)
    anomaly_rate = anomaly_count / total_count * 100
    
    print(f"Model: {model_name}")
    print(f"Residual Std: {std_residual:.2f}, Threshold: {threshold_value:.2f}")
    print(f"异常数量: {anomaly_count}/{total_count} ({anomaly_rate:.1f}%)")

    return df_dashboard