import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Area, AreaChart, ReferenceLine,
} from 'recharts';
import {
  AlertTriangle, Activity, TrendingUp,
  Database, Settings, RefreshCw,
} from 'lucide-react';

const API_BASE = 'http://localhost:5000'; // 后端 Flask 地址

const NetworkTrafficDashboard = () => {
  const [selectedModel, setSelectedModel] = useState('NF01_2_l-32_d-0.1');
  const [modelList, setModelList] = useState([]);
  const [data, setData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({
    mae: 0, rmse: 0, mape: 0, anomalyCount: 0,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [threshold, setThreshold] = useState(2.5); // σ 倍数
  const [errorMsg, setErrorMsg] = useState('');

  // ---------- 从后端拉模型列表 ----------
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/models`);
        const json = await res.json();
        if (json.models && json.models.length > 0) {
          setModelList(json.models);

          // 如果当前选中的不在列表里，就默认第一个
          if (!json.models.includes(selectedModel)) {
            setSelectedModel(json.models[0]);
          }
        }
      } catch (err) {
        console.error('获取模型列表失败：', err);
        setErrorMsg('无法获取模型列表，请检查后端是否启动。');
      }
    };

    fetchModels();
  }, []);

  // ---------- 核心：调用后端拿真实预测 ----------
  const fetchDataFromBackend = async (modelName, thr) => {
    setIsLoading(true);
    setErrorMsg('');

    try {
      // limit=200 就是最近 200 个点，你可以按需改
      const res = await fetch(
        `${API_BASE}/api/predict/${encodeURIComponent(modelName)}?limit=200`,
      );
      const json = await res.json();

      if (json.error) {
        setErrorMsg(json.error);
        setData([]);
        setAlerts([]);
        setStats({ mae: 0, rmse: 0, mape: 0, anomalyCount: 0 });
        setIsLoading(false);
        return;
      }

      const raw = json.data || [];

      // === 在前端根据阈值计算残差/异常等 ===
      const processed = [];
      const alertsTmp = [];
      let anomalyCount = 0;

      const sigmaBase = 50; // 你原来 mock 里是 threshold * 50，这里就沿用

      raw.forEach((item, idx) => {
        const minute = item.minute;
        const actual = Number(item.actual);
        const predicted = Number(item.predicted);
        const residual = actual - predicted;
        const upperBound = predicted + thr * sigmaBase;
        const lowerBound = predicted - thr * sigmaBase;
        const isAnomaly = Math.abs(residual) > thr * sigmaBase;

        if (isAnomaly) {
          anomalyCount += 1;
          alertsTmp.push({
            id: idx,
            minute,
            time: `minute_index ${minute}`,
            severity: Math.abs(residual) > thr * sigmaBase * 1.6 ? 'high' : 'medium',
            message: `异常流量检测: 偏差 ${residual.toFixed(0)} flows`,
            actual: actual.toFixed(0),
            predicted: predicted.toFixed(0),
          });
        }

        processed.push({
          minute,
          actual,
          predicted,
          residual,
          upperBound,
          lowerBound,
          isAnomaly,
        });
      });

      // 统计指标
      if (processed.length > 0) {
        const mae =
          processed.reduce((sum, d) => sum + Math.abs(d.residual), 0)
          / processed.length;
        const rmse = Math.sqrt(
          processed.reduce((sum, d) => sum + d.residual ** 2, 0) / processed.length,
        );
        const mape =
          processed.reduce(
            (sum, d) => sum + (Math.abs(d.residual / d.actual) * 100 || 0),
            0,
          ) / processed.length;

        setStats({
          mae,
          rmse,
          mape,
          anomalyCount,
        });
      } else {
        setStats({ mae: 0, rmse: 0, mape: 0, anomalyCount: 0 });
      }

      setData(processed);
      setAlerts(alertsTmp.slice(-10).reverse()); // 最近 10 条

    } catch (err) {
      console.error('获取预测数据失败：', err);
      setErrorMsg('获取预测数据失败，请检查后端是否启动或网络连接。');
      setData([]);
      setAlerts([]);
      setStats({ mae: 0, rmse: 0, mape: 0, anomalyCount: 0 });
    }

    setIsLoading(false);
  };

  // ---------- 当模型或阈值变化时，重新拉数据 ----------
  useEffect(() => {
    if (!selectedModel) return;
    fetchDataFromBackend(selectedModel, threshold);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel, threshold]);

  const handleRefresh = () => {
    if (!selectedModel) return;
    fetchDataFromBackend(selectedModel, threshold);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* Header */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">网络流量预测预警系统</h1>
              <p className="text-slate-300">基于 LSTM 的实时流量监控与异常检测</p>
              {errorMsg && (
                <p className="mt-2 text-sm text-red-400">
                  {errorMsg}
                </p>
              )}
            </div>
            <button
              onClick={handleRefresh}
              disabled={isLoading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              {isLoading ? '刷新中...' : '刷新数据'}
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard
            icon={<Activity className="w-6 h-6" />}
            title="平均绝对误差"
            value={stats.mae.toFixed(2)}
            unit="flows"
            color="blue"
          />
          <StatCard
            icon={<TrendingUp className="w-6 h-6" />}
            title="均方根误差"
            value={stats.rmse.toFixed(2)}
            unit="flows"
            color="green"
          />
          <StatCard
            icon={<Database className="w-6 h-6" />}
            title="平均百分比误差"
            value={stats.mape.toFixed(2)}
            unit="%"
            color="purple"
          />
          <StatCard
            icon={<AlertTriangle className="w-6 h-6" />}
            title="异常点数量"
            value={stats.anomalyCount}
            unit="个"
            color="red"
          />
        </div>

        {/* Model Selection & Settings */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-blue-500/20">
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <Settings className="w-5 h-5 text-slate-400" />
              <span className="text-slate-300">模型选择:</span>
            </div>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-4 py-2 bg-slate-700 text-white rounded-lg border border-slate-600 focus:outline-none focus:border-blue-500"
            >
              {modelList.length === 0 ? (
                <option value="">暂无模型</option>
              ) : (
                modelList.map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))
              )}
            </select>

            <div className="flex items-center gap-2 ml-auto">
              <span className="text-slate-300">异常阈值:</span>
              <input
                type="number"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value) || 1)}
                step="0.1"
                min="1"
                max="5"
                className="w-20 px-3 py-2 bg-slate-700 text-white rounded-lg border border-slate-600 focus:outline-none focus:border-blue-500"
              />
              <span className="text-slate-400 text-sm">σ</span>
            </div>
          </div>
        </div>

        {/* Main Chart */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20">
          <h2 className="text-xl font-bold text-white mb-4">流量预测 vs 实际值</h2>
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={data}>
              <defs>
                <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="minute"
                stroke="#94a3b8"
                label={{
                  value: 'Minute Index',
                  position: 'insideBottom',
                  offset: -5,
                  fill: '#94a3b8',
                }}
              />
              <YAxis
                stroke="#94a3b8"
                label={{
                  value: 'Flows',
                  angle: -90,
                  position: 'insideLeft',
                  fill: '#94a3b8',
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #3b82f6',
                  borderRadius: '8px',
                  color: '#fff',
                }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="upperBound"
                stroke="none"
                fill="#ef4444"
                fillOpacity={0.1}
                name="异常上界"
              />
              <Area
                type="monotone"
                dataKey="lowerBound"
                stroke="none"
                fill="#ef4444"
                fillOpacity={0.1}
                name="异常下界"
              />
              <Area
                type="monotone"
                dataKey="actual"
                stroke="#3b82f6"
                strokeWidth={2}
                fill="url(#colorActual)"
                name="实际流量"
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                name="预测流量"
              />
              {data.filter((d) => d.isAnomaly).map((point, idx) => (
                <ReferenceLine
                  key={idx}
                  x={point.minute}
                  stroke="#ef4444"
                  strokeWidth={2}
                  strokeDasharray="3 3"
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Residual Chart */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20">
          <h2 className="text-xl font-bold text-white mb-4">残差分析</h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="minute" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #3b82f6',
                  borderRadius: '8px',
                  color: '#fff',
                }}
              />
              <ReferenceLine y={0} stroke="#64748b" strokeWidth={2} />
              <Line
                type="monotone"
                dataKey="residual"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={(props) => {
                  const { cx, cy, payload } = props;
                  if (payload.isAnomaly) {
                    return (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={4}
                        fill="#ef4444"
                        stroke="#fff"
                        strokeWidth={2}
                      />
                    );
                  }
                  return null;
                }}
                name="残差"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Alerts Panel */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <AlertTriangle className="w-6 h-6 text-red-500" />
            异常预警列表
          </h2>
          <div className="space-y-2">
            {alerts.length === 0 ? (
              <div className="text-center py-8 text-slate-400">
                暂无异常检测
              </div>
            ) : (
              alerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`p-4 rounded-lg border-l-4 ${
                    alert.severity === 'high'
                      ? 'bg-red-900/20 border-red-500'
                      : 'bg-yellow-900/20 border-yellow-500'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <AlertTriangle
                          className={`w-4 h-4 ${
                            alert.severity === 'high'
                              ? 'text-red-500'
                              : 'text-yellow-500'
                          }`}
                        />
                        <span className="font-semibold text-white">
                          {alert.message}
                        </span>
                      </div>
                      <p className="text-sm text-slate-300">
                        时间: {alert.time} | 实际: {alert.actual} | 预测: {alert.predicted}
                      </p>
                    </div>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        alert.severity === 'high'
                          ? 'bg-red-500 text-white'
                          : 'bg-yellow-500 text-slate-900'
                      }`}
                    >
                      {alert.severity === 'high' ? '高危' : '中危'}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

const StatCard = ({
  icon, title, value, unit, color,
}) => {
  const colorClasses = {
    blue: 'from-blue-600 to-blue-700',
    green: 'from-green-600 to-green-700',
    purple: 'from-purple-600 to-purple-700',
    red: 'from-red-600 to-red-700',
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-blue-500/20">
      <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${colorClasses[color]} mb-3`}>
        {icon}
      </div>
      <p className="text-slate-400 text-sm mb-1">{title}</p>
      <p className="text-2xl font-bold text-white">
        {value}
        {' '}
        <span className="text-sm text-slate-400">{unit}</span>
      </p>
    </div>
  );
};

export default NetworkTrafficDashboard;
