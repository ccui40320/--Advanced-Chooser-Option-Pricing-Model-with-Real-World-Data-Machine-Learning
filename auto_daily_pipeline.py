"""
Week 2 — 自动化数据预处理与特征工程流水线 (增强版)
===================================================
功能:
  1. 动态拉取全量数据 (2018-01-01 ~ 今天)
  2. 数据清洗 (缺失值插值, IQR 异常值截断, 时间对齐)
  3. 特征工程 (传统 + 高级 + 情绪)
  4. 自动保存为 Dynamic_Cleaned_Dataset.csv

数据源:
  - Yahoo Finance: JPM 股价/股息, VIX 指数
  - FRED: 国债利率 DGS3MO, DGS1, DGS10
  - NewsAPI / GDELT: 金融新闻情绪 (可选)

输出特征 (≥15 个):
  传统: Daily_Return, Volatility_21D, Dividend_Yield, Dividend_Growth
  高级: VIX_JPM_Corr, Interest_Rate_Momentum, VIX_Return
  情绪: Sentiment_Score (混合模型), Article_Count
  BSM 就绪: RiskFreeRate (小数), Dividend_Yield (年化), Volatility_21D
"""

import pandas as pd
import numpy as np
import os
import warnings
import datetime

import yfinance as yf
from fredapi import Fred

warnings.filterwarnings('ignore')

# ============================================================
# 0. 环境与密钥配置
# ============================================================

NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '5bcd874aaa5b44c3856ace2eac805e13')
FRED_API_KEY = os.environ.get('FRED_API_KEY', '20b77339750fc4721483e772416e9841')

IS_GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS') == 'true'

if IS_GITHUB_ACTIONS:
    print("[环境] GitHub 云端 — 海外直连")
    PROXY_PORT = None
else:
    PROXY_PORT = '7892'  # 龙猫云代理端口
    print(f"[环境] 本地调试 — 代理端口 {PROXY_PORT}")
    os.environ['HTTP_PROXY'] = f'http://127.0.0.1:{PROXY_PORT}'
    os.environ['HTTPS_PROXY'] = f'http://127.0.0.1:{PROXY_PORT}'

START_DATE = '2018-01-01'
TODAY_STR = datetime.date.today().strftime('%Y-%m-%d')


# ============================================================
# 1. 动态获取全量基础数据
# ============================================================

def fetch_all_market_data():
    """获取 JPM 股价/股息、VIX 指数、FRED 多期限利率"""
    print(f"\n[1/4] 拉取全量市场数据 ({START_DATE} ~ {TODAY_STR})...")

    # --- 1a. JPM ---
    print("  -> JPM 行情...")
    jpm = yf.Ticker("JPM")
    df_jpm = jpm.history(start=START_DATE, end=TODAY_STR, auto_adjust=False)
    df_div = jpm.dividends
    df_div = df_div[(df_div.index >= START_DATE)]

    df_market = df_jpm[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
    df_market.columns = ['JPM_Open', 'JPM_High', 'JPM_Low', 'JPM_Close',
                         'JPM_Adj_Close', 'JPM_Volume']
    df_market['JPM_Dividend'] = df_div.reindex(df_market.index).fillna(0)

    # --- 1b. VIX ---
    print("  -> VIX 恐慌指数...")
    vix = yf.Ticker("^VIX")
    df_vix = vix.history(start=START_DATE, end=TODAY_STR)[['Close']]
    df_vix.columns = ['VIX_Close']

    # 去时区
    df_market.index = df_market.index.tz_localize(None)
    df_vix.index = df_vix.index.tz_localize(None)

    # --- 1c. FRED 多期限利率 (3M, 1Y, 10Y) ---
    print("  -> FRED 国债利率 (DGS3MO, DGS1, DGS10)...")
    fred_series = {
        'RiskFreeRate_3M': 'DGS3MO',
        'RiskFreeRate_1Y': 'DGS1',
        'RiskFreeRate_10Y': 'DGS10',
    }
    df_rates = pd.DataFrame(index=df_market.index)
    try:
        fred = Fred(api_key=FRED_API_KEY)
        for col_name, code in fred_series.items():
            rates = fred.get_series(code, observation_start=START_DATE)
            df_rates[col_name] = rates.reindex(df_market.index).ffill()
            print(f"    {col_name} ({code}): OK")
    except Exception as e:
        print(f"    FRED 警告: {e}")

    # 合并
    raw_df = df_market.join(df_vix, how='left').join(df_rates, how='left')
    print(f"  -> 原始数据: {len(raw_df)} 行, {len(raw_df.columns)} 列")
    return raw_df


# ============================================================
# 2. 真实 NLP 情感 (NewsAPI, 最近 28 天)
# ============================================================

def fetch_real_nlp_sentiment():
    """从 NewsAPI 获取近期新闻并做 NLP 情感分析 (仅最近 28 天)"""
    print("\n[2/4] NLP 情感模块 (NewsAPI, 最近 28 天)...")
    import requests
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # 确保 NLTK 数据已下载
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

    url = "https://newsapi.org/v2/everything"
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=28)

    params = {
        'q': 'JPMorgan OR JPM',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': NEWS_API_KEY
    }

    try:
        proxies = None if IS_GITHUB_ACTIONS else {'http': f'http://127.0.0.1:{PROXY_PORT}',
                                                   'https': f'http://127.0.0.1:{PROXY_PORT}'}
        resp = requests.get(url, params=params, proxies=proxies, timeout=15)
        data = resp.json()

        if data.get('status') != 'ok':
            print(f"  NewsAPI: {data.get('message', 'unknown error')}")
            return pd.Series(dtype=float)

        articles = data.get('articles', [])
        sia = SentimentIntensityAnalyzer()
        records = []

        for a in articles:
            text = str(a.get('title', '')) + '. ' + str(a.get('description', ''))
            pub_date = a.get('publishedAt', '')[:10]
            score = sia.polarity_scores(text)['compound']
            records.append({'Date': pub_date, 'Score': (score + 1) / 2})

        if not records:
            return pd.Series(dtype=float)

        df_news = pd.DataFrame(records)
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        daily = df_news.groupby('Date')['Score'].mean()
        print(f"  -> NLP 完成: {len(daily)} 天真实情感得分")
        return daily
    except Exception as e:
        print(f"  -> NLP 模块: {e}")
        return pd.Series(dtype=float)


# ============================================================
# 3. 数据清洗与特征工程
# ============================================================

def build_features(df, real_nlp_series):
    """
    清洗 + 特征工程流水线。
    包括: 缺失值插值, IQR 截断, 传统特征, 高级特征, 情绪混合模型
    """
    print("\n[3/4] 数据清洗与特征工程...")

    df = df.sort_index()
    # 仅保留交易日
    df = df[df.index.dayofweek < 5]

    # --- 3a. 缺失值处理 ---
    interp_cols = ['JPM_Open', 'JPM_High', 'JPM_Low', 'JPM_Close',
                   'JPM_Adj_Close', 'VIX_Close', 'RiskFreeRate_3M',
                   'RiskFreeRate_1Y', 'RiskFreeRate_10Y']
    interp_cols = [c for c in interp_cols if c in df.columns]
    df[interp_cols] = df[interp_cols].interpolate(method='time', limit_direction='both')
    df['JPM_Volume'] = df['JPM_Volume'].ffill()

    # --- 3b. IQR 异常值截断 ---
    def cap_outliers(series, mult=3.0):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return np.clip(series, Q1 - mult * IQR, Q3 + mult * IQR)

    df['JPM_Volume'] = cap_outliers(df['JPM_Volume'])
    df['VIX_Close'] = cap_outliers(df['VIX_Close'])

    # --- 3c. 传统特征 ---
    # 对数收益率 (基于 Adj_Close, 考虑分红和拆股)
    df['Daily_Return'] = np.log(df['JPM_Adj_Close'] / df['JPM_Adj_Close'].shift(1))

    # 21 天滚动年化波动率 (更标准, 21 = 一个月交易日)
    df['Volatility_21D'] = df['Daily_Return'].rolling(window=21).std() * np.sqrt(252)

    # 年化股息率 q (滚动 252 天累计分红 / 当前股价)
    div_12m = df['JPM_Dividend'].rolling(window=252, min_periods=1).sum()
    df['Dividend_Yield'] = div_12m / df['JPM_Close']
    df['Dividend_Yield'] = df['Dividend_Yield'].ffill().fillna(0)

    # 股息增长率
    df['Dividend_Growth'] = (df['JPM_Dividend'].rolling(window=252).sum()).pct_change(
        periods=252).fillna(0)

    # --- 3d. 高级特征 ---
    # 利率: 百分比 → 小数 (BSM 格式)
    for c in ['RiskFreeRate_3M', 'RiskFreeRate_1Y', 'RiskFreeRate_10Y']:
        if c in df.columns:
            df[f'{c}_Decimal'] = df[c] / 100.0

    # VIX 日收益率
    df['VIX_Return'] = df['VIX_Close'].pct_change()

    # VIX-JPM 滚动相关性 (捕捉市场体制转换)
    df['VIX_JPM_Corr'] = df['Daily_Return'].rolling(window=21).corr(df['VIX_Return'])

    # 利率动量 (5日 vs 20日均线差)
    df['Rate_MA5'] = df['RiskFreeRate_3M_Decimal'].rolling(window=5).mean()
    df['Rate_MA20'] = df['RiskFreeRate_3M_Decimal'].rolling(window=20).mean()
    df['Interest_Rate_Momentum'] = df['Rate_MA5'] - df['Rate_MA20']

    # --- 3e. 混合情绪模型 ---
    # 历史基底: VIX 倒置 (低 VIX = 高情绪) + 收益率信号
    vix_norm = (df['VIX_Close'] - df['VIX_Close'].min()) / (
        df['VIX_Close'].max() - df['VIX_Close'].min() + 1e-6)
    vix_sentiment = 1 - vix_norm  # VIX 越低 -> 情绪越高

    ret_sent = (df['Daily_Return'].rolling(21).mean() + 0.02) / 0.04  # 收益信号
    df['Sentiment_Score'] = (vix_sentiment * 0.5 + ret_sent.clip(0, 1) * 0.5)
    df['Sentiment_Score'] = df['Sentiment_Score'].ffill().fillna(0.5)

    # 真实 NLP 覆盖 (仅覆盖最近有 NewsAPI 数据的日期)
    if not real_nlp_series.empty:
        overlap = df.index.intersection(real_nlp_series.index)
        if len(overlap) > 0:
            # NewsAPI 权重: 0.6 + 历史代理权重: 0.4 (平滑过渡)
            nlp_aligned = real_nlp_series.reindex(overlap)
            historical = df.loc[overlap, 'Sentiment_Score']
            df.loc[overlap, 'Sentiment_Score'] = nlp_aligned * 0.6 + historical * 0.4
            print(f"  -> NLP 融合: {len(overlap)} 天覆盖 (权重 0.6)")

    # 情绪数据计数 (NewsAPI 有数据的天数为 1)
    df['Article_Count'] = 0
    if not real_nlp_series.empty:
        overlap = df.index.intersection(real_nlp_series.index)
        df.loc[overlap, 'Article_Count'] = 1

    # 删掉开头 NaN 行 (收益率 / 波动率第一天算不出来)
    df = df.dropna(subset=['Daily_Return', 'Volatility_21D'])
    # 但保留第一行的空值在后续列中
    df = df.dropna(subset=[c for c in ['JPM_Close', 'VIX_Close', 'RiskFreeRate_3M_Decimal']
                           if c in df.columns])

    print(f"  -> 特征构建完成: {len(df)} 行, {len(df.columns)} 列")
    return df


# ============================================================
# 4. 主程序
# ============================================================

if __name__ == "__main__":
    try:
        print("=" * 55)
        print("  Quant Pipeline: 数据预处理与特征工程")
        print(f"  日期: {START_DATE} ~ {TODAY_STR}")
        print("=" * 55)

        # 1. 获取全量市场数据
        print("\n[START] Step 1: fetch_all_market_data")
        raw = fetch_all_market_data()

        # 2. 获取 NLP 情感 (最近 28 天)
        print("\n[START] Step 2: fetch_real_nlp_sentiment")
        nlp = fetch_real_nlp_sentiment()

        # 3. 清洗 + 特征工程
        print("\n[START] Step 3: build_features")
        if not raw.empty:
            final = build_features(raw, nlp)

            # 4. 保存
            output = 'Dynamic_Cleaned_Dataset.csv'
            final.to_csv(output)

            print(f"\n[4/4] 完成!")
            print(f"  文件: {output}")
            print(f"  维度: {final.shape[0]} 行 x {final.shape[1]} 列")
            print(f"  范围: {final.index[0].date()} ~ {final.index[-1].date()}")

            # 打印 BSM 参数就绪状态
            bsm_cols = {
                'S0': 'JPM_Close',
                'r (小数)': 'RiskFreeRate_3M_Decimal',
                'q (年化股息率)': 'Dividend_Yield',
                'sigma (21D波动率)': 'Volatility_21D',
                'VIX': 'VIX_Close',
                '情绪': 'Sentiment_Score',
            }
            print("\n  BSM 参数:")
            for name, col in bsm_cols.items():
                ok = col in final.columns
                print(f"    {'[OK]' if ok else '[--]'} {name:15s} {col}")
        else:
            print("\n[FAIL] 数据获取失败")
    except Exception as e:
        print(f"\n[PIPELINE ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
