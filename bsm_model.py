"""
BSM Chooser Option Pricing Model (增强版)
=========================================
基于 Rubinstein (1991) 的选择性期权定价公式。

功能:
  - BSM Call/Put 定价 (含股息率 q)
  - Chooser Option 定价 (Call + Put 分解)
  - Greeks (Delta, Gamma, Vega, Theta, Rho)
  - 向量化批量定价
  - 真实数据加载与定价

BSM 参数:
  S     = 标的资产价格 (JPM_Close)
  K     = 行权价 (strike)
  T     = 到期时间 (年)
  r     = 无风险利率 (小数)
  sigma = 波动率
  q     = 年化股息率 (小数)

Chooser Option (Rubinstein):
  Chooser = Call(S, K, T2) + Put(S, K·exp(-r·(T2-T1)), T1)
  其中 T1 = 选择日, T2 = 最终到期日
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import json
import os


# ============================================================
# 1. BSM 基础函数
# ============================================================

def d1_func(S, K, T, r, sigma, q=0.0):
    """BSM d1 统计量"""
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2_func(S, K, T, r, sigma, q=0.0):
    """BSM d2 统计量"""
    return d1_func(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bsm_call(S, K, T, r, sigma, q=0.0):
    """
    BSM Call 期权定价 (含连续股息率 q)
    返回: 期权价格
    """
    if T <= 0 or sigma <= 0:
        return np.maximum(S - K, 0) if T <= 0 else 0.0

    d1 = d1_func(S, K, T, r, sigma, q)
    d2 = d2_func(S, K, T, r, sigma, q)

    price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def bsm_put(S, K, T, r, sigma, q=0.0):
    """
    BSM Put 期权定价 (含连续股息率 q)
    返回: 期权价格
    """
    if T <= 0 or sigma <= 0:
        return np.maximum(K - S, 0) if T <= 0 else 0.0

    d1 = d1_func(S, K, T, r, sigma, q)
    d2 = d2_func(S, K, T, r, sigma, q)

    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price


# ============================================================
# 2. Greeks (风险指标)
# ============================================================

def bsm_delta(S, K, T, r, sigma, q=0.0, option_type='call'):
    """Delta: 标的价格变动 1 单位时期权价格的变化"""
    d1 = d1_func(S, K, T, r, sigma, q)
    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1)


def bsm_gamma(S, K, T, r, sigma, q=0.0):
    """Gamma: Delta 对标的价格的敏感度"""
    d1 = d1_func(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bsm_vega(S, K, T, r, sigma, q=0.0):
    """Vega: 波动率变动 1% 时期权价格的变化"""
    d1 = d1_func(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100


def bsm_theta(S, K, T, r, sigma, q=0.0, option_type='call'):
    """Theta: 时间流逝 1 天时期权价格的变化 (日均)"""
    d1 = d1_func(S, K, T, r, sigma, q)
    d2 = d2_func(S, K, T, r, sigma, q)
    days = 365

    if option_type == 'call':
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
    return theta / days


def bsm_rho(S, K, T, r, sigma, q=0.0, option_type='call'):
    """Rho: 利率变动 1% 时期权价格的变化"""
    d2 = d2_func(S, K, T, r, sigma, q)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def compute_greeks(S, K, T, r, sigma, q=0.0):
    """同时计算 Call 和 Put 的所有 Greeks"""
    result = {}
    for opt_type in ['call', 'put']:
        result[f'{opt_type}_delta'] = bsm_delta(S, K, T, r, sigma, q, opt_type)
        result[f'{opt_type}_theta'] = bsm_theta(S, K, T, r, sigma, q, opt_type)
        result[f'{opt_type}_rho'] = bsm_rho(S, K, T, r, sigma, q, opt_type)

    # Gamma 和 Vega 对 call/put 相同
    result['gamma'] = bsm_gamma(S, K, T, r, sigma, q)
    result['vega'] = bsm_vega(S, K, T, r, sigma, q)
    return result


# ============================================================
# 3. Chooser Option 定价 (核心)
# ============================================================

def chooser_price(S, K, T1, T2, r, sigma, q=0.0):
    """
    Rubinstein 选择性期权定价公式:
      Chooser = Call(S, K, T2) + Put(S, K·exp(-r·(T2-T1)), T1)

    参数:
      S     : 当前股价
      K     : 行权价
      T1    : 选择日 (年)
      T2    : 到期日 (年), T2 > T1
      r     : 无风险利率 (小数)
      sigma : 年化波动率
      q     : 年化股息率 (小数)

    返回:
      price : Chooser 期权价格
      breakdown: {call_component, put_component} 用于分析
    """
    # Call 部分: 标准 BSM Call 到 T2
    call_part = bsm_call(S, K, T2, r, sigma, q)

    # Put 部分: 调整行权价 K_adj = K * exp(-r * (T2 - T1))
    # 这是 Rubinstein 公式的核心: 在 T1 时, 持有者选择 Call 或 Put
    # Put 的有效行权价需要折现到 T1
    K_adj = K * np.exp(-r * (T2 - T1))
    put_part = bsm_put(S, K_adj, T1, r, sigma, q)

    price = call_part + put_part

    return price, {
        'call_component': call_part,
        'put_component': put_part,
        'K_adj': K_adj,
        'call_weight': call_part / price if price > 0 else 0,
    }


def chooser_greeks(S, K, T1, T2, r, sigma, q=0.0):
    """
    Chooser Option 的 Greeks (通过线性组合计算)
    Chooser = Call(T2) + Put(T1 with adjusted K)
    """
    d1_call = d1_func(S, K, T2, r, sigma, q)
    d2_call = d2_func(S, K, T2, r, sigma, q)

    K_adj = K * np.exp(-r * (T2 - T1))
    d1_put = d1_func(S, K_adj, T1, r, sigma, q)
    d2_put = d2_func(S, K_adj, T1, r, sigma, q)

    # Delta
    delta = (np.exp(-q * T2) * norm.cdf(d1_call) +
             np.exp(-q * T1) * (norm.cdf(d1_put) - 1))

    # Gamma (相同, 都是 d1 的导数)
    gamma = (np.exp(-q * T2) * norm.pdf(d1_call) / (S * sigma * np.sqrt(T2)) +
             np.exp(-q * T1) * norm.pdf(d1_put) / (S * sigma * np.sqrt(T1)))

    # Vega
    vega = (S * np.exp(-q * T2) * norm.pdf(d1_call) * np.sqrt(T2) / 100 +
            S * np.exp(-q * T1) * norm.pdf(d1_put) * np.sqrt(T1) / 100)

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
    }


# ============================================================
# 4. 向量化定价 (批量处理)
# ============================================================

def price_chooser_vectorized(S_array, K, T1, T2, r_array, sigma_array, q_array=None):
    """
    向量化批量定价 — 输入可以是数组/Series
    用于对历史数据集进行批量定价
    """
    S_arr = np.asarray(S_array, dtype=float)
    r_arr = np.asarray(r_array, dtype=float)
    sigma_arr = np.asarray(sigma_array, dtype=float)

    if q_array is not None:
        q_arr = np.asarray(q_array, dtype=float)
    else:
        q_arr = np.zeros_like(S_arr)

    prices = []
    breakdowns = []
    for i in range(len(S_arr)):
        p, b = chooser_price(S_arr[i], K, T1, T2, r_arr[i], sigma_arr[i], q_arr[i])
        prices.append(p)
        breakdowns.append(b)

    return np.array(prices), breakdowns


# ============================================================
# 5. 真实数据加载与定价
# ============================================================

def load_real_data(csv_path=None):
    """
    加载 Week 1/2 生成的真实数据集。
    自动搜索可能的位置。
    """
    if csv_path is None:
        # 自动搜索
        search_paths = [
            '../week2_ Deliverables/Dynamic_Cleaned_Dataset.csv',
            '../week1_ Deliverables/Project_Data_JPM_2018_2024.csv',
            'Dynamic_Cleaned_Dataset.csv',
            'Project_Data_JPM_2018_2024.csv',
        ]
        for path in search_paths:
            if os.path.exists(path):
                csv_path = path
                break

    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True, encoding='utf-8')
        print(f"[OK] 加载数据: {csv_path} ({len(df)} 行)")
        return df
    else:
        print("[WARN] 未找到真实数据集，使用示例数据")
        return None


def price_from_real_data(df, K=150.0, T1=0.5, T2=1.0):
    """
    用真实数据集批量计算 Chooser 期权价格。
    自动从 DataFrame 中查找所需列。

    所需列:
      - JPM_Close  (S)
      - RiskFreeRate_3M_Decimal 或 RiskFreeRate_3M (r)
      - Volatility_21D 或 Rolling_Vol_20D (sigma)
      - Dividend_Yield 或 从 JPM_Dividend 推导 (q)
    """
    # 自动匹配列名
    price_col = [c for c in ['JPM_Close', 'Close'] if c in df.columns][0]

    rate_col = [c for c in ['RiskFreeRate_3M_Decimal', 'RiskFreeRate_3M', 'risk_free_rate']
                if c in df.columns][0]
    rate_is_pct = rate_col == 'RiskFreeRate_3M'  # True if still in percentage

    vol_col = [c for c in ['Volatility_21D', 'Rolling_Vol_20D', 'sigma']
               if c in df.columns][0]

    div_col = [c for c in ['Dividend_Yield', 'dividend_yield', 'JPM_Dividend']
               if c in df.columns][0]

    S = df[price_col].values
    r_raw = df[rate_col].values
    r = r_raw / 100.0 if rate_is_pct else r_raw
    sigma = df[vol_col].values

    if 'Dividend_Yield' in df.columns or 'dividend_yield' in df.columns:
        q = df[div_col].values
    else:
        # 从 JPM_Dividend 推导 (简化: 年化累计/价格)
        div_12m = df[div_col].rolling(252, min_periods=1).sum()
        q = (div_12m / df[price_col]).ffill().fillna(0).values

    # 过滤 NaN
    valid = ~(np.isnan(S) | np.isnan(r) | np.isnan(sigma))
    S, r, sigma, q = S[valid], r[valid], sigma[valid], q[valid]

    prices, breakdowns = price_chooser_vectorized(S, K, T1, T2, r, sigma, q)

    result = pd.DataFrame({
        'Date': df.index[valid],
        'S': S,
        'r': r,
        'sigma': sigma,
        'q': q,
        'Chooser_Price': prices,
        'Call_Component': [b['call_component'] for b in breakdowns],
        'Put_Component': [b['put_component'] for b in breakdowns],
    })
    result = result.dropna()
    return result


# ============================================================
# 6. 配置文件管理
# ============================================================

DEFAULT_CONFIG = {
    "stock_price": 239.32,         # JPM 最新收盘价 (2024-12-30)
    "strike_price": 150.0,         # 行权价 (匹配论文)
    "risk_free_rate": 0.0437,      # 3个月国债利率 (小数)
    "sigma": 0.1868,               # 21天年化波动率
    "dividend_yield": 0.0240,      # 年化股息率
    "T1": 0.5,                     # 选择日 (半年)
    "T2": 1.0,                     # 到期日 (1年)
    "source": "JPM real data (2024-12-30)",
    "bsm_params_ready": {
        "S0": "JPM_Close",
        "r": "RiskFreeRate_3M_Decimal",
        "q": "Dividend_Yield",
        "sigma": "Volatility_21D"
    }
}


def load_config(config_path='config.json'):
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"[WARN] {config_path} not found, using defaults")
    return DEFAULT_CONFIG.copy()


def save_config(config, config_path='config.json'):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"[OK] Config saved: {config_path}")


# ============================================================
# 7. Demo / 测试入口
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  BSM Chooser Option Model — 测试")
    print("=" * 50)

    # 加载配置
    config = load_config()
    S = config['stock_price']
    K = config['strike_price']
    T1 = config['T1']
    T2 = config['T2']
    r = config['risk_free_rate']
    sigma = config['sigma']
    q = config['dividend_yield']

    print(f"\n参数: S={S}, K={K}, T1={T1}, T2={T2}")
    print(f"      r={r:.4f}, sigma={sigma:.4f}, q={q:.4f}")

    # Chooser 定价
    price, breakdown = chooser_price(S, K, T1, T2, r, sigma, q)
    print(f"\nChooser 期权价格: {price:.4f}")
    print(f"  Call 部分: {breakdown['call_component']:.4f}")
    print(f"  Put 部分:  {breakdown['put_component']:.4f}")

    # Greeks
    greeks = chooser_greeks(S, K, T1, T2, r, sigma, q)
    print(f"\nChooser Greeks:")
    for k, v in greeks.items():
        print(f"  {k}: {v:.6f}")
