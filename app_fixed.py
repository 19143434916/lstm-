#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM流量监控系统后端
支持WebSocket实时通信
"""

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np
import json
import time
import threading
import random
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量
monitoring_active = False
monitoring_thread = None
current_interval = 3  # 默认3秒更新间隔

class TrafficMonitor:
    def __init__(self):
        self.anomaly_count = 0
        self.base_traffic = 15000
        self.time_step = 0
        
    def generate_traffic_data(self):
        """生成模拟的流量数据"""
        # 添加时间周期性变化
        self.time_step += 1
        time_factor = np.sin(self.time_step * 0.1) * 2000
        
        # 基础流量 + 随机波动 + 时间因素
        actual = self.base_traffic + random.gauss(0, 1000) + time_factor
        
        # LSTM预测值（模拟）- 略有偏差
        predicted = actual + random.gauss(0, 300)
        
        # 计算边界
        std_dev = 2000  # 标准差
        upper_bound = predicted + 2 * std_dev
        lower_bound = predicted - 2 * std_dev
        
        # 判断是否异常（5%概率产生异常）
        is_anomaly = False
        if random.random() < 0.05:  # 5%概率产生异常
            # 产生异常流量
            if random.random() < 0.5:
                actual = predicted + random.uniform(3000, 8000)  # 高流量异常
            else:
                actual = predicted - random.uniform(3000, 8000)  # 低流量异常
            is_anomaly = True
            self.anomaly_count += 1
        
        # 计算准确率
        error = abs(actual - predicted)
        accuracy = max(0, 100 - (error / actual * 100))
        
        return {
            'timestamp': datetime.now().isoformat(),
            'actual': actual,
            'predicted': predicted,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'is_anomaly': is_anomaly,
            'accuracy': accuracy,
            'anomaly_count': self.anomaly_count
        }

monitor = TrafficMonitor()

def monitoring_loop():
    """监控循环"""
    global monitoring_active, current_interval
    
    while monitoring_active:
        try:
            # 生成数据
            data = monitor.generate_traffic_data()
            
            # 发送到所有连接的客户端
            socketio.emit('traffic_update', data, namespace='/')
            
            # 等待指定间隔
            time.sleep(current_interval)
            
        except Exception as e:
            print(f"监控循环错误: {e}")
            break

@app.route('/')
def index():
    """主页路由"""
    return send_from_directory('.', 'index_fixed.html')

@socketio.on('connect')
def handle_connect():
    """处理客户端连接"""
    print('客户端已连接')
    emit('connected', {'message': '连接成功'})

@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开"""
    print('客户端已断开')

@socketio.on('message')
def handle_message(message):
    """处理客户端消息"""
    global monitoring_active, monitoring_thread, current_interval
    
    try:
        data = json.loads(message) if isinstance(message, str) else message
        action = data.get('action')
        
        if action == 'start':
            if not monitoring_active:
                monitoring_active = True
                current_interval = data.get('interval', 3)
                monitor.anomaly_count = 0  # 重置异常计数
                
                # 启动监控线程
                monitoring_thread = threading.Thread(target=monitoring_loop)
                monitoring_thread.daemon = True
                monitoring_thread.start()
                
                emit('status', {'message': '监控已启动', 'status': 'running'})
                print(f'监控已启动，更新间隔: {current_interval}秒')
        
        elif action == 'pause':
            monitoring_active = False
            emit('status', {'message': '监控已暂停', 'status': 'paused'})
            print('监控已暂停')
        
        elif action == 'stop':
            monitoring_active = False
            monitor.anomaly_count = 0
            emit('status', {'message': '监控已停止', 'status': 'stopped'})
            print('监控已停止')
    
    except Exception as e:
        print(f"处理消息错误: {e}")
        emit('error', {'message': str(e)})

# WebSocket路由（用于原生WebSocket连接）
@socketio.on('connect', namespace='/ws')
def handle_ws_connect():
    """处理WebSocket连接"""
    print('WebSocket客户端已连接')

@socketio.on('message', namespace='/ws')
def handle_ws_message(message):
    """处理WebSocket消息"""
    handle_message(message)

if __name__ == '__main__':
    print("="*50)
    print("LSTM流量监控系统后端")
    print("="*50)
    print("服务器启动在: http://localhost:5000")
    print("WebSocket端点: ws://localhost:5000/ws")
    print("请在浏览器中打开 http://localhost:5000 查看界面")
    print("="*50)
    
    # 启动服务器
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)