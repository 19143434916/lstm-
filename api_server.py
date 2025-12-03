# api_server.py
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from nf01_utils import run_model_and_build_df, MODEL_DIR  # 你之前的工具里已经有 MODEL_DIR

app = Flask(__name__)
CORS(app)  # 允许前端跨域访问（React 在 3000 端口）

# ---------- 工具函数：列出可用模型 ----------
def list_available_models():
    models = []
    for fn in os.listdir(MODEL_DIR):
        if fn.endswith(".h5") or fn.endswith(".keras"):
            name = os.path.splitext(fn)[0]  # 去掉后缀
            models.append(name)
    models.sort()
    return models


# ---------- 1) 获取模型列表 ----------
@app.route("/api/models", methods=["GET"])
def get_models():
    models = list_available_models()
    return jsonify({"models": models})


# ---------- 2) 获取某个模型的预测结果 ----------
@app.route("/api/predict/<model_name>", methods=["GET"])
def get_predictions(model_name):
    """
    返回格式：
    [
      {
        "minute": 40123,
        "actual": 27890.12,
        "predicted": 27654.33
      },
      ...
    ]
    """
    # 先检查模型是否存在
    available = list_available_models()
    if model_name not in available:
        return jsonify({
            "error": f"模型 {model_name} 不存在，可用模型：{available}"
        }), 400

    # 可选：limit 参数，控制返回多少个点（默认 200）
    try:
        limit = int(request.args.get("limit", 200))
    except ValueError:
        limit = 200

    # 调用你写好的工具，在测试集上跑一遍真实预测
    df_pred = run_model_and_build_df(model_name, save_csv=False)

    # df_pred 的 index 是 minute_index，列是 y_true, y_pred, residual
    df_pred = df_pred.reset_index()  # 把 minute_index 变成普通列

    # 只取最后 limit 个点
    if limit > 0:
        df_pred = df_pred.tail(limit)

    records = []
    for _, row in df_pred.iterrows():
        records.append({
            "minute": int(row["minute_index"]),
            "actual": float(row["y_true"]),
            "predicted": float(row["y_pred"]),
            # 如果你想直接拿后端算好的残差也可以一起丢给前端
            "residual": float(row["residual"]),
        })

    return jsonify({
        "model": model_name,
        "count": len(records),
        "data": records,
    })


if __name__ == "__main__":
    # 这里的 host/port 按需改，这样前端用 http://localhost:5000 就能访问
    app.run(host="0.0.0.0", port=5000, debug=True)
