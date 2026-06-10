"""
API 连通性测试脚本
测试 Yahoo Finance (yfinance), FRED (fredapi), Alpha Vantage 三个数据源

使用方法:
    python API_test.py              # 使用代理 (默认为 127.0.0.1:7892)
    python API_test.py --no-proxy   # 不使用代理 (直连)
"""

import os
import sys
import argparse

# ===== 配置区域 =====

# 代理端口 (用于解决 Yahoo 封锁 IP 问题)
PROXY_PORT = '7892'
PROXY_URL = f'http://127.0.0.1:{PROXY_PORT}'

# FRED API Key
FRED_API_KEY = '20b77339750fc4721483e772416e9841'

# Alpha Vantage API Key
ALPHA_VANTAGE_KEY = 'Q26PJUM322QP4MWE'

# ===== 修复 Windows GBK 编码 =====
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('gbk', 'gb2312'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ===== 参数解析 =====

parser = argparse.ArgumentParser(description='API连通性测试')
parser.add_argument('--no-proxy', action='store_true', help='不使用代理直连')
args = parser.parse_args()

# ===== 代理设置 =====

if not args.no_proxy:
    os.environ['HTTP_PROXY'] = PROXY_URL
    os.environ['HTTPS_PROXY'] = PROXY_URL
    print(f"已配置系统代理: {PROXY_URL}")
    print("  (若连接失败，可尝试 python API_test.py --no-proxy)")
else:
    print("未配置代理 (直连模式)")
print()


# ===== API 测试函数 =====

import yfinance as yf
import pandas as pd
from fredapi import Fred
from alpha_vantage.timeseries import TimeSeries


def _get_close(data):
    """
    兼容 yfinance 1.2.0 (MultiIndex 列) 和旧版。
    返回收盘价的 Series。
    """
    if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
        return data['Close'].iloc[:, 0]
    return data['Close']


def test_yahoo_jpm():
    """测试 Yahoo Finance — JPM 股价"""
    print("[TEST] Yahoo Finance (JPM)...")
    try:
        jpm = yf.download("JPM", start="2024-01-01", end="2024-01-10", progress=False)
        if jpm is not None and not jpm.empty:
            close_series = _get_close(jpm)
            print(f"  [OK] 获取 {len(jpm)} 行数据 | 最新价: {close_series.iloc[-1]:.2f}")
            return True
        else:
            print("  [WARN] 数据为空")
            return False
    except Exception as e:
        msg = str(e)
        if 'rate' in msg.lower() or 'too many' in msg.lower():
            print("  [FAIL] 频率限制 (Rate Limited)。请检查代理。")
        else:
            print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def test_yahoo_vix():
    """测试 Yahoo Finance — VIX 指数"""
    print("[TEST] Yahoo Finance (^VIX)...")
    try:
        vix = yf.download("^VIX", start="2024-01-01", end="2024-01-10", progress=False)
        if vix is not None and not vix.empty:
            close_series = _get_close(vix)
            print(f"  [OK] 获取 {len(vix)} 行数据 | 最新VIX: {close_series.iloc[-1]:.2f}")
            return True
        else:
            print("  [WARN] 数据为空")
            return False
    except Exception as e:
        msg = str(e)
        if 'rate' in msg.lower() or 'too many' in msg.lower():
            print("  [FAIL] 频率限制 (Rate Limited)。请检查代理。")
        else:
            print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def test_fred():
    """测试 FRED — 国债利率 (DGS3MO, DGS1, DGS10)"""
    print("[TEST] FRED (国债利率)...")
    series_list = {
        'DGS3MO': '3个月国债收益率',
        'DGS1': '1年期国债收益率',
        'DGS10': '10年期国债收益率',
    }
    all_ok = True
    try:
        fred = Fred(api_key=FRED_API_KEY)
        for code, name in series_list.items():
            try:
                rate = fred.get_series(code, observation_start='2024-01-01',
                                       observation_end='2024-01-10')
                count = len(rate.dropna())
                if count > 0:
                    print(f"  [OK] {code} ({name}): {count} 个有效值")
                else:
                    print(f"  [WARN] {code} ({name}): 数据为空")
                    all_ok = False
            except Exception as e:
                print(f"  [FAIL] {code} ({name}): {type(e).__name__}: {e}")
                all_ok = False
        return all_ok
    except Exception as e:
        print(f"  [FAIL] FRED 连接失败: {type(e).__name__}: {e}")
        return False


def test_alpha_vantage():
    """测试 Alpha Vantage — JPM 数据"""
    print("[TEST] Alpha Vantage (JPM)...")
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        av_data, meta = ts.get_daily(symbol='JPM', outputsize='compact')
        if not av_data.empty:
            print(f"  [OK] 获取 {len(av_data)} 行数据")
            return True
        else:
            print("  [WARN] 数据为空")
            return False
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def run_all_tests():
    """运行所有 API 测试并生成报告"""
    print("=" * 50)
    print("  API 连通性测试开始")
    print("=" * 50)
    print()

    results = {
        'Yahoo Finance (JPM)': test_yahoo_jpm(),
        'Yahoo Finance (VIX)': test_yahoo_vix(),
        'FRED (国债利率)': test_fred(),
        'Alpha Vantage (JPM)': test_alpha_vantage(),
    }

    print()
    print("=" * 50)
    print("  测试报告汇总")
    print("=" * 50)
    for source, status in results.items():
        icon = "[PASS]" if status else "[FAIL]"
        print(f"  {icon} {source}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  结果: {passed}/{total} 通过")

    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
