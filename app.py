from flask import Flask, jsonify, send_file
from flask_cors import CORS
import threading
import time
from datetime import datetime
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ============ 尝试导入LSTM模型 ============
try:
    from nf01_utils import get_dashboard_data
    USE_LSTM_MODEL = True
    print("✅ 成功加载 LSTM 模型")
except ImportError:
    USE_LSTM_MODEL = False
    print("⚠️ 未加载nf01_utils，使用模拟数据")

# ============ 全局数据 ============
current_data = {
    'total_kb': 0.0, 'kb_sent': 0.0, 'kb_recv': 0.0,
    'is_anomaly': False, 'anomaly_score': 0.0, 'predictions': [],
    'timestamp': '', 'actual': 0.0, 'predicted': 0.0,
    'residual': 0.0, 'upperBound': 0.0, 'lowerBound': 0.0
}

lstm_cache = []
lstm_idx = 0

def load_lstm_data():
    global lstm_cache, USE_LSTM_MODEL
    if USE_LSTM_MODEL:
        try:
            df = get_dashboard_data("NF01_2_l-32_d-0.1", threshold_sigma=2.5)
            print(f"📋 DataFrame列名: {df.columns.tolist()}")
            lstm_cache = df.to_dict('records')
            print(f"✅ 加载 {len(lstm_cache)} 条数据")
            if lstm_cache:
                print(f"📊 示例数据: {lstm_cache[0]}")
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            print("⚠️ 切换到模拟数据模式")
            USE_LSTM_MODEL = False
            lstm_cache = []

def gen_sim_data():
    base = 1000 + np.sin(time.time() * 0.1) * 200
    actual = max(0, base + np.random.normal(0, 50))
    predicted = base + np.random.normal(0, 20)
    residual = actual - predicted
    threshold = 2.5 * 80
    return {
        'actual': round(actual, 2), 'predicted': round(predicted, 2),
        'residual': round(residual, 2), 'upperBound': round(predicted + threshold, 2),
        'lowerBound': round(max(0, predicted - threshold), 2),
        'isAnomaly': abs(residual) > threshold
    }

def monitor_thread():
    global current_data, lstm_idx
    if USE_LSTM_MODEL: load_lstm_data()
    
    while True:
        try:
            if USE_LSTM_MODEL and lstm_cache:
                d = lstm_cache[lstm_idx % len(lstm_cache)]
                lstm_idx += 1
                # 兼容不同的列名
                actual_val = float(d.get('actual', d.get('y_true', 0)))
                predicted_val = float(d.get('predicted', d.get('y_pred', 0)))
                residual_val = float(d.get('residual', actual_val - predicted_val))
                current_data.update({
                    'total_kb': actual_val,
                    'kb_sent': actual_val * 0.4,
                    'kb_recv': actual_val * 0.6,
                    'is_anomaly': bool(d.get('isAnomaly', False)),
                    'anomaly_score': abs(residual_val) / 80 if residual_val else 0,
                    'predictions': [predicted_val],
                    'actual': actual_val,
                    'predicted': predicted_val,
                    'residual': residual_val,
                    'upperBound': float(d.get('upperBound', predicted_val + 200)),
                    'lowerBound': float(d.get('lowerBound', max(0, predicted_val - 200))),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                d = gen_sim_data()
                current_data.update({
                    'total_kb': d['actual'], 'kb_sent': d['actual'] * 0.4,
                    'kb_recv': d['actual'] * 0.6, 'is_anomaly': d['isAnomaly'],
                    'anomaly_score': abs(d['residual']) / 80,
                    'predictions': [d['predicted']], 'actual': d['actual'],
                    'predicted': d['predicted'], 'residual': d['residual'],
                    'upperBound': d['upperBound'], 'lowerBound': d['lowerBound'],
                    'timestamp': datetime.now().isoformat()
                })
            time.sleep(2)
        except Exception as e:
            print(f"❌ 错误: {e}")
            time.sleep(5)

@app.route('/')
def index():
    # 直接返回HTML文件
    return send_file('index.html')

@app.route('/api/current_data')
def get_current_data():
    return jsonify(current_data)

@app.route('/api/generate_report')
def generate_report():
    return jsonify({
        'status': 'success', 'data_points': len(lstm_cache) or 100,
        'avg_upload_kb': round(current_data['kb_sent'], 2),
        'avg_download_kb': round(current_data['kb_recv'], 2),
        'peak_traffic_kb': round(current_data['total_kb'] * 1.5, 2),
        'std_traffic_kb': 80.0, 'monitoring_duration_min': 60,
        'current_time': datetime.now().isoformat()
    })

if __name__ == '__main__':
    threading.Thread(target=monitor_thread, daemon=True).start()
    print("=" * 50)
    print("🚀 LSTM网络流量预警系统")
    print(f"📊 模式: {'LSTM模型' if USE_LSTM_MODEL else '模拟数据'}")
    print("🌐 地址: http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)