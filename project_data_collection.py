"""
Week 1 — 数据采集脚本 (增强版)
=================================
采集 JPM (2018-2024) 的金融、宏观及情绪数据，为 Chooser Option 定价模型准备。

数据源:
  - Yahoo Finance: JPM 股价/股息, VIX 指数
  - Alpha Vantage: JPM 股价 (备用/补充)
  - FRED: 国债利率 (DGS3MO, DGS1, DGS10)
  - NewsAPI: 金融新闻情绪 (可选，需要 API Key)

输出: Project_Data_JPM_2018_2024.csv (1760+ 行, 15+ 列)

BSM 参数就绪:
  - S0 (股价): JPM_Close [OK]
  - r (利率): RiskFreeRate_Xyr (已转为小数格式) [OK]
  - q (股息率): Dividend_Yield (年化) [OK]
  - σ (波动率): Daily_Return (用于后续计算) [OK]
"""

import os
import sys
import warnings
import time
from datetime import datetime

# 修复 Windows GBK 编码 (避免 emoji 报错)
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('gbk', 'gb2312'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred

warnings.filterwarnings('ignore')

# ============================================================
# 1. 配置区域
# ============================================================

# [代理设置] 解决 Yahoo 封锁问题
PROXY_PORT = '7892'
proxy_url = f'http://127.0.0.1:{PROXY_PORT}'
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

# [API Keys]
FRED_API_KEY = '20b77339750fc4721483e772416e9841'
ALPHA_VANTAGE_KEY = 'Q26PJUM322QP4MWE'

# 情绪数据: GDELT (免费, 无需Key) 或 NewsAPI (需要Key)
USE_SENTIMENT = 'gdelt'       # 'gdelt' → 免费, 无需Key; 'newsapi' → 需要Key; '' → 跳过
NEWSAPI_KEY = ''               # 仅 USE_SENTIMENT='newsapi' 时需要

# [时间范围]
START_DATE = '2018-01-01'
END_DATE = '2024-12-31'


# ============================================================
# 2. 辅助函数
# ============================================================

def safe_api_call(func, desc="API调用", max_retries=2):
    """带重试机制的 API 调用包装器"""
    for attempt in range(max_retries + 1):
        try:
            result = func()
            return result
        except Exception as e:
            if attempt < max_retries:
                print(f"  [RETRY] {desc} 第{attempt+1}次失败，重试中... ({e})")
                time.sleep(2 ** attempt)
            else:
                print(f"  [FAIL] {desc} 失败: {type(e).__name__}: {e}")
                return None


def annualize_dividend_yield(dividends, prices, window=252):
    """
    计算年化股息率 q。
    使用滚动12个月的总分红 / 当前股价。
    若窗口内无分红，则返回 NaN (后续用前值填充)。

    参数:
        dividends: Series, 每日分红金额 (无分红日为 0)
        prices: Series, 每日股价
        window: 滚动窗口天数 (252 = 1个交易日年)
    返回:
        Series, 年化股息率 (小数形式)
    """
    # 滚动12个月累计分红
    div_12m = dividends.rolling(window=window, min_periods=1).sum()
    # 股息率 = 累计分红 / 股价
    div_yield = div_12m / prices
    # 用前值填充空值 (数据开头可能无历史分红数据)
    div_yield = div_yield.ffill().fillna(0)
    return div_yield


# ============================================================
# 3. 数据采集函数
# ============================================================

def collect_financial_data():
    """
    从 Yahoo Finance 采集 JPM 和 VIX 数据。
    返回: (df_jpm, df_vix) 或 (None, None) 失败时
    """
    print("\n[模块 A] Yahoo Finance 数据采集")
    print("-" * 40)

    # --- 3a. JPM 股价 ---
    print("[A1/4] 正在获取 JPM 股价与股息...")
    try:
        jpm = yf.Ticker("JPM")
        # auto_adjust=False 拿到原始价格
        jpm_hist = jpm.history(start=START_DATE, end=END_DATE, auto_adjust=False)

        if jpm_hist.empty:
            print("  [FAIL] JPM 数据为空")
            return None, None
        print(f"  [OK] JPM 股价: {len(jpm_hist)} 行")

        # 股息
        jpm_divs = jpm.dividends
        jpm_divs = jpm_divs[(jpm_divs.index >= START_DATE) &
                            (jpm_divs.index <= END_DATE)]
        print(f"  [OK] JPM 股息: {len(jpm_divs)} 笔记录")
    except Exception as e:
        print(f"  [FAIL] JPM 数据获取失败: {e}")
        return None, None

    # --- 3b. VIX 指数 ---
    print("[A2/4] 正在获取 VIX 指数...")
    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(start=START_DATE, end=END_DATE)
        if vix_hist.empty:
            print("  [WARN]  VIX 数据为空")
            vix_hist = None
        else:
            print(f"  [OK] VIX: {len(vix_hist)} 行")
    except Exception as e:
        print(f"  [FAIL] VIX 获取失败: {e}")
        vix_hist = None

    # --- 3c. Alpha Vantage (备用数据源) ---
    print("[A3/4] 正在从 Alpha Vantage 获取 JPM (备用)...")
    av_hist = None
    try:
        from alpha_vantage.timeseries import TimeSeries
        ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        av_data, _ = ts.get_daily(symbol='JPM', outputsize='compact')
        if not av_data.empty:
            # Alpha Vantage 索引是字符串 '2024-01-02'，转为 datetime
            av_data.index = pd.to_datetime(av_data.index)
            # 筛选时间范围
            mask = (av_data.index >= START_DATE) & (av_data.index <= END_DATE)
            av_hist = av_data.loc[mask].sort_index()
            print(f"  [OK] Alpha Vantage: {len(av_hist)} 行")
        else:
            print("  [WARN]  Alpha Vantage 数据为空")
    except ImportError:
        print("  [WARN]  alpha_vantage 未安装，跳过")
    except Exception as e:
        msg = str(e)
        if 'premium' in msg.lower():
            print("  [WARN]  Alpha Vantage free版仅支持compact模式，跳过全量对比")
        else:
            print(f"  [WARN]  Alpha Vantage 获取失败: {e}")

    # --- 3d. 数据清洗与合并准备 ---
    print("[A4/4] 正在清洗与整理数据...")

    # 去除时区信息
    jpm_hist.index = jpm_hist.index.tz_localize(None)

    # 构建 JPM DataFrame
    df_jpm = jpm_hist[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
    df_jpm.columns = ['JPM_Open', 'JPM_High', 'JPM_Low', 'JPM_Close',
                       'JPM_Adj_Close', 'JPM_Volume']

    # 合并股息
    df_jpm['JPM_Dividend'] = jpm_divs.reindex(df_jpm.index).fillna(0)

    # === 新特征: 年化股息率 q ===
    df_jpm['Dividend_Yield'] = annualize_dividend_yield(
        df_jpm['JPM_Dividend'], df_jpm['JPM_Close']
    )

    # === 新特征: 日收益率 (用于后续波动率计算) ===
    df_jpm['Daily_Return'] = df_jpm['JPM_Close'].pct_change()

    # === 新特征: 对数收益率 (更精确的波动率计算) ===
    df_jpm['Log_Return'] = np.log(df_jpm['JPM_Close'] / df_jpm['JPM_Close'].shift(1))

    # === 新特征: 历史波动率 (21天滚动, 年化) ===
    df_jpm['Volatility_21D'] = df_jpm['Log_Return'].rolling(window=21).std() * np.sqrt(252)

    # 处理 VIX
    if vix_hist is not None:
        vix_hist.index = vix_hist.index.tz_localize(None)
        df_vix = vix_hist[['Close']].copy()
        df_vix.columns = ['VIX_Close']
    else:
        df_vix = pd.DataFrame(index=jpm_hist.index)

    # Alpha Vantage 对比 (放在数据清洗之后，确保日期已对齐)
    if av_hist is not None and len(av_hist) > 0:
        av_hist.index = av_hist.index.tz_localize(None)
        common_dates = df_jpm.index.intersection(av_hist.index)
        if len(common_dates) > 0:
            yahoo_close = df_jpm.loc[common_dates, 'JPM_Close']
            av_close = av_hist.loc[common_dates, '4. close']
            diff_pct = (yahoo_close - av_close).abs() / yahoo_close
            print(f"  [CHECK] Yahoo vs Alpha Vantage (重叠 {len(common_dates)} 天): "
                  f"平均差异={diff_pct.mean():.4%}, 最大差异={diff_pct.max():.4%}")
            if diff_pct.max() > 0.05:
                print("  [WARN]  两个数据源差异较大，建议检查数据质量")
        else:
            print("  [CHECK] Yahoo与Alpha Vantage无重叠日期，跳过对比")

    return df_jpm, df_vix


def collect_macro_data():
    """
    从 FRED 采集宏观经济数据。
    包括: DGS3MO (3个月), DGS1 (1年), DGS10 (10年) 国债利率。
    利率已转换为小数形式 (如 1.44% -> 0.0144)。

    返回: DataFrame 或 None
    """
    print("\n[模块 B] FRED 宏观数据采集")
    print("-" * 40)

    fred_series = {
        'RiskFreeRate_3M': 'DGS3MO',   # 3个月期国债收益率
        'RiskFreeRate_1Y': 'DGS1',     # 1年期国债收益率
        'RiskFreeRate_10Y': 'DGS10',   # 10年期国债收益率
    }

    try:
        fred = Fred(api_key=FRED_API_KEY)

        df_rates = pd.DataFrame()
        for col_name, series_code in fred_series.items():
            print(f"  [B] 正在获取 {col_name} ({series_code})...", end=" ")
            rates = fred.get_series(series_code,
                                    observation_start=START_DATE,
                                    observation_end=END_DATE)
            if not rates.empty:
                # === 关键修复: 百分比 → 小数 ===
                # FRED 返回的数据是百分比形式 (如 5.50 表示 5.50%)
                # BSM 公式要求小数形式 (如 0.0550)
                df_rates[col_name] = rates / 100.0
                print(f"[OK] {len(rates.dropna())} 个有效值")
            else:
                print("[WARN]  空数据")
                df_rates[col_name] = np.nan

        return df_rates

    except Exception as e:
        print(f"  [FAIL] FRED 数据获取失败: {type(e).__name__}: {e}")
        return None


def collect_sentiment_gdelt():
    """
    [方案A] 使用 GDELT Project 采集金融新闻情绪 (推荐)
     - 完全免费，无需 API Key
     - 覆盖 1979 年至今的全球新闻
     - 自带 Tone 情感评分
     - 按季度采样，避免请求过于频繁

    返回: DataFrame (Date, Sentiment_Score, Article_Count) 或 None
    """
    print("\n[模块 C - 方案A] GDELT 金融新闻情绪采集")
    print("  (免费, 无需API Key, 覆盖 2018-2024)")
    print("-" * 40)

    try:
        from gdeltdoc import GdeltDoc, Filters
    except ImportError:
        print("  [WARN] gdeltdoc 未安装: pip install gdeltdoc")
        return None

    gd = GdeltDoc()

    # 按年查询 (GDELT 单次查询范围不宜过大)
    years = range(2018, 2025)
    all_records = []

    for year in years:
        start = f'{year}-01-01'
        end = f'{year}-12-31'
        try:
            f = Filters(
                keyword='JPMorgan JPM banking finance',
                start_date=start,
                end_date=end,
                num_records=50  # 每年取 50 篇代表性新闻
            )
            articles = gd.article_search(f)

            if articles is not None and not articles.empty:
                # GDELT 返回的 tone 列是情感评分 (范围约 -100 ~ 100)
                # 归一化到 [0, 1]: score = (tone/100 + 1) / 2
                tones = articles['tone'].values
                titles = articles.get('title', [''] * len(articles))
                date_strs = articles['seendate'].values

                # 按日期聚合并计算月度平均 sentiment
                for i in range(len(tones)):
                    try:
                        tone = float(tones[i])
                        # 归一化到 [0, 1]
                        sentiment = max(0, min(1, (tone / 100.0 + 1) / 2))
                        # GDELT 日期格式: YYYYMMDDHHMMSS
                        ds = str(date_strs[i])
                        date = pd.to_datetime(ds[:8], format='%Y%m%d')
                        all_records.append({
                            'Date': date,
                            'Sentiment_Score': sentiment,
                        })
                    except (ValueError, IndexError):
                        continue

                print(f"  [OK] {year}: {len(articles)} 篇新闻")
                time.sleep(1)  # GDELT 限速保护
            else:
                print(f"  [WARN] {year}: 无数据")

        except Exception as e:
            print(f"  [WARN] {year}: {e}")
            time.sleep(2)

    if not all_records:
        return None

    # 按日聚合后重采样为月度数据
    df = pd.DataFrame(all_records)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # 按月计算平均情绪
    monthly = df.resample('ME').agg({
        'Sentiment_Score': 'mean',
    })
    monthly['Article_Count'] = df.resample('ME').size()

    print(f"\n  [OK] GDELT 情绪数据: {len(monthly)} 个月")
    print(f"  情绪范围: [{monthly['Sentiment_Score'].min():.3f}, "
          f"{monthly['Sentiment_Score'].max():.3f}]")

    return monthly


def collect_sentiment_newsapi(api_key):
    """
    [方案B] 使用 NewsAPI 采集金融新闻情绪 (备选)
    需要有效的 NewsAPI Key，免费版仅限最近 1 个月数据。
    """
    print("\n[模块 C - 方案B] NewsAPI 情绪数据采集")
    print("-" * 40)

    try:
        from newsapi import NewsApiClient
        from textblob import TextBlob
    except ImportError:
        print("  [WARN] 需要安装: pip install newsapi-python textblob")
        return None

    newsapi = NewsApiClient(api_key=api_key)

    # 因免费版限制，只拉最近几个月的
    end = pd.Timestamp(END_DATE)
    start = end - pd.DateOffset(months=6)
    sentiment_records = []

    print(f"  [B] 采集 {start.date()} ~ {end.date()} 新闻情绪...")

    try:
        articles = newsapi.get_everything(
            q='JPMorgan OR JPM',
            from_param=start.strftime('%Y-%m-%d'),
            to=end.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy',
            page_size=50
        )
        if articles['status'] == 'ok' and articles['articles']:
            for art in articles['articles']:
                text = (art['title'] or '') + ' ' + (art['description'] or '')
                if text.strip():
                    blob = TextBlob(text)
                    pol = blob.sentiment.polarity  # [-1, 1]
                    sentiment_records.append({
                        'Date': pd.to_datetime(art['publishedAt'][:10]),
                        'Sentiment_Score': (pol + 1) / 2,  # → [0, 1]
                    })
        if sentiment_records:
            df = pd.DataFrame(sentiment_records)
            df['Date'] = pd.to_datetime(df['Date'])
            monthly = df.set_index('Date').resample('ME').mean()
            monthly['Article_Count'] = df.set_index('Date').resample('ME').size()
            print(f"  [OK] 新闻情绪: {len(monthly)} 个月")
            return monthly
    except Exception as e:
        print(f"  [FAIL] NewsAPI: {e}")

    return None


def collect_sentiment_proxy(df_jpm, df_vix):
    """
    [方案C] 基于市场数据的情绪代理 (自动备用)
    当外部API都不可用时，用 VIX + 收益率推导情绪。

    逻辑: VIX 低 + 正收益 → 乐观; VIX 高 + 负收益 → 悲观
    """
    print("\n[模块 C - 方案C] 基于市场数据的情绪代理 (自动备用)")

    result = df_jpm[[]].copy()

    # 特征1: 从 VIX 推导恐慌情绪 (VIX越高越恐慌)
    if 'VIX_Close' in result.columns or df_vix is not None:
        vix = df_vix['VIX_Close'] if df_vix is not None else df_jpm.get('VIX_Close')
        if vix is not None:
            # VIX < 15 → 低恐慌 → 情绪偏正; VIX > 30 → 高恐慌 → 情绪偏负
            vix_sentiment = 1 - (vix - vix.min()) / (vix.max() - vix.min() + 1e-6)
            result['Sentiment_VIX'] = vix_sentiment.clip(0, 1)

    # 特征2: 从日收益率推导 (正收益→乐观, 负收益→悲观)
    returns = df_jpm['Daily_Return']
    if returns is not None:
        # 用21天滚动平均收益率作为情绪代理
        ret_sentiment = (returns.rolling(21).mean() + 0.1) / 0.2  # 假设日均收益 ~0%
        result['Sentiment_Return'] = ret_sentiment.clip(0, 1)

    # 综合: 等权平均
    sent_cols = [c for c in result.columns if c.startswith('Sentiment_')]
    if sent_cols:
        result['Sentiment_Score'] = result[sent_cols].mean(axis=1)
        result['Sentiment_Score'] = result['Sentiment_Score'].ffill().fillna(0.5)
        print(f"  [OK] 情绪代理已生成 ({len(sent_cols)} 个特征融合)")
        monthly = result[['Sentiment_Score']].resample('ME').mean()
        monthly['Article_Count'] = 0
        return monthly

    return None


def collect_sentiment_data(mode, df_jpm=None, df_vix=None, newsapi_key=None):
    """
    情绪数据采集入口。
    按 mode 选择方案: 'gdelt' / 'newsapi' / '' (跳过)

    返回: DataFrame 或 None
    """
    if not mode:
        print("\n[模块 C] 情绪数据: 已跳过 (USE_SENTIMENT='')")
        return None

    result = None

    if mode == 'gdelt':
        result = collect_sentiment_gdelt()
    elif mode == 'newsapi' and newsapi_key:
        result = collect_sentiment_newsapi(newsapi_key)

    # 如果外部API都失败了, 用市场数据做情绪代理
    if result is None and df_jpm is not None:
        print("  → 外部API不可用，使用市场数据情绪代理")
        result = collect_sentiment_proxy(df_jpm, df_vix)

    if result is not None:
        print(f"  [OK] 情绪数据就绪: {len(result)} 期")
    else:
        print("  [WARN] 无法生成情绪数据")

    return result


# ============================================================
# 4. 数据整合与主程序
# ============================================================

def merge_datasets(df_jpm, df_vix, df_rates, df_sentiment=None):
    """
    以 JPM 交易日为基准，整合所有数据。
    处理缺失值 (前向填充)。
    """
    print("\n[合并] 正在整合所有数据...")

    # 以 JPM 为基准
    final = df_jpm.copy()

    # 合并 VIX
    if df_vix is not None and not df_vix.empty:
        final = final.join(df_vix, how='left')

    # 合并利率
    if df_rates is not None:
        final = final.join(df_rates, how='left')

    # 合并情绪 (如果有)
    if df_sentiment is not None:
        # 情绪是季度数据，前向填充到每日
        final = final.join(df_sentiment, how='left')
        final['Sentiment_Score'] = final['Sentiment_Score'].ffill()
        final['Article_Count'] = final['Article_Count'].fillna(0).astype(int)

    # === 缺失值处理 ===
    # VIX 和利率在非交易日无数据，使用前向填充
    cols_to_ffill = ['VIX_Close']
    if df_rates is not None:
        cols_to_ffill += list(df_rates.columns)

    for col in cols_to_ffill:
        if col in final.columns:
            final[col] = final[col].ffill()

    # 删除开头的 NaN — 只对关键列做检查 (股价/VIX/利率不能为空)
    # Daily_Return, Log_Return, Volatility_21D 开头天然是 NaN, 不参与检查
    essential_cols = ['JPM_Close', 'VIX_Close']
    if df_rates is not None:
        essential_cols += list(df_rates.columns)
    essential_cols = [c for c in essential_cols if c in final.columns]
    final = final.dropna(subset=essential_cols)

    # 确保情绪列无空值
    if 'Sentiment_Score' in final.columns:
        final['Sentiment_Score'] = final['Sentiment_Score'].fillna(0.5)

    return final


def main():
    """主程序入口"""
    print("=" * 55)
    print("  Week 1 数据采集任务 (增强版)")
    print(f"  时间范围: {START_DATE} ~ {END_DATE}")
    print(f"  执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    # Step 1: Yahoo Finance + Alpha Vantage
    df_jpm, df_vix = collect_financial_data()
    if df_jpm is None:
        print("\n[FAIL] 严重错误: 无法获取 JPM 数据，程序终止。")
        sys.exit(1)

    # Step 2: FRED 宏观数据
    df_rates = collect_macro_data()

    # Step 3: 情绪数据 (可选)
    # 默认用 gdelt (免费, 无需Key), 也可改成 'newsapi' 或 '' 跳过
    df_sentiment = collect_sentiment_data(
        mode=USE_SENTIMENT,
        df_jpm=df_jpm,
        df_vix=df_vix,
        newsapi_key=NEWSAPI_KEY
    )

    # Step 4: 数据整合
    final_df = merge_datasets(df_jpm, df_vix, df_rates, df_sentiment)

    # Step 5: 保存文件
    output_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
    output_file = os.path.join(output_dir, 'Project_Data_JPM_2018_2024.csv')
    final_df.to_csv(output_file)

    # ===== 输出报告 =====
    print("\n" + "=" * 55)
    print("  [OK] 数据采集完成!")
    print("=" * 55)
    print(f"  保存路径: {output_file}")
    print(f"  数据维度: {final_df.shape[0]} 行 × {final_df.shape[1]} 列")
    print(f"  时间范围: {final_df.index[0].strftime('%Y-%m-%d')} ~ "
          f"{final_df.index[-1].strftime('%Y-%m-%d')}")

    print("\n  列清单:")
    for col in final_df.columns:
        non_null = final_df[col].notna().sum()
        print(f"    {col:25s}  {non_null:6d}/{len(final_df)} 非空")

    print("\n  BSM 定价参数状态:")
    bsm_params = {
        'S0 (股价)': 'JPM_Close' in final_df.columns,
        'r (无风险利率)': any(c.startswith('RiskFreeRate') for c in final_df.columns),
        'q (年化股息率)': 'Dividend_Yield' in final_df.columns,
        'σ (波动率基础)': 'Daily_Return' in final_df.columns,
        'VIX (市场波动)': 'VIX_Close' in final_df.columns,
    }
    for param, ready in bsm_params.items():
        print(f"    {'[OK]' if ready else '[FAIL]'} {param}")

    print(f"\n  数据预览 (前3行):")
    print(final_df.head(3).to_string())

    return final_df


if __name__ == "__main__":
    df = main()
