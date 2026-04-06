import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def features(df):
    """
    全新 18 特徵組（基於五篇高引用論文）
    來源：arXiv 2311.14759 / 2410.06935 / 2511.00665 / MDPI TFT / GitHub baruch1192
    分類：原始價量(2) + 趨勢(4) + 動量(5) + 波動(3) + 量能(4)
    """
    print("🛠️ 開始計算全新 18 特徵組 (基於五篇高引用論文)...")
    df = df.copy()

    # ── 共用 True Range 計算（供 ADX、ATR 使用）─────────
    high_low  = df['High'] - df['Low']
    high_prev = (df['High'] - df['Close'].shift(1)).abs()
    low_prev  = (df['Low']  - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)

    # ==========================================
    # 📌 1. 趨勢類（4 個）
    # 論文：arXiv 2511.00665 — 最佳 EMA 6/95；ADX(13)
    # ==========================================
    df['EMA_6']     = df['Close'].ewm(span=6,  adjust=False).mean()
    df['EMA_95']    = df['Close'].ewm(span=95, adjust=False).mean()
    df['EMA_Cross'] = df['EMA_6'] - df['EMA_95']   # 黃金/死亡交叉強度

    # ADX(13)：趨勢強弱指標
    plus_dm  = (df['High'] - df['High'].shift(1)).clip(lower=0)
    minus_dm = (df['Low'].shift(1) - df['Low']).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr13    = tr.ewm(span=13, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=13, adjust=False).mean()  / (atr13 + 1e-8)
    minus_di = 100 * minus_dm.ewm(span=13, adjust=False).mean() / (atr13 + 1e-8)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['ADX_13'] = dx.ewm(span=13, adjust=False).mean()

    # ==========================================
    # 📌 2. 動量類（5 個）
    # 論文：arXiv 2410.06935 Chi-Squared Top3：RSI30、MACD、MOM30
    #       arXiv 2511.00665：MACD(17, 21, 15)
    # ==========================================
    # RSI(14)
    d14   = df['Close'].diff()
    g14   = d14.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    l14   = (-d14.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + g14 / (l14 + 1e-8)))

    # RSI(30)
    d30   = df['Close'].diff()
    g30   = d30.clip(lower=0).ewm(alpha=1/30, adjust=False).mean()
    l30   = (-d30.clip(upper=0)).ewm(alpha=1/30, adjust=False).mean()
    df['RSI_30'] = 100 - (100 / (1 + g30 / (l30 + 1e-8)))

    # MACD(17, 21) + Signal(15)
    ema17 = df['Close'].ewm(span=17, adjust=False).mean()
    ema21 = df['Close'].ewm(span=21, adjust=False).mean()
    df['MACD']        = ema17 - ema21
    df['MACD_Signal'] = df['MACD'].ewm(span=15, adjust=False).mean()

    # MOM(30)：動能
    df['MOM_30'] = df['Close'] - df['Close'].shift(30)

    # ==========================================
    # 📌 3. 波動類（3 個）
    # ==========================================
    df['ATR_14']  = tr.rolling(14).mean()

    sma20  = df['Close'].rolling(20).mean()
    std20  = df['Close'].rolling(20).std()
    upper  = sma20 + 2 * std20
    lower  = sma20 - 2 * std20
    df['BB_Width'] = (upper - lower) / (sma20 + 1e-8) * 100
    df['BB_PB']    = (df['Close'] - lower) / (upper - lower + 1e-8)

    # ==========================================
    # 📌 4. 量能類（4 個）
    # 論文：MDPI TFT — 主動買量最具預測力
    #       GitHub baruch1192 — CMF 取代 MFI
    # ==========================================
    # OBV
    obv_vals = np.where(df['Close'] > df['Close'].shift(1),  df['Volume'],
               np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    df['OBV'] = pd.Series(obv_vals, index=df.index).cumsum()

    # CMF(14)：柴金資金流量（比 MFI 更純粹）
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) \
          / (df['High'] - df['Low'] + 1e-8)
    df['CMF_14'] = (clv * df['Volume']).rolling(14).sum() \
                   / (df['Volume'].rolling(14).sum() + 1e-8)

    # Volume_Ratio：相對成交量（過濾異常放量）
    df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)

    # Taker_Ratio：主動買方比例（Binance 獨有，反映多空力道）
    if 'Taker_buy_quote_asset_volume' in df.columns \
            and 'Quote_asset_volume' in df.columns:
        df['Taker_Ratio'] = df['Taker_buy_quote_asset_volume'] \
                            / (df['Quote_asset_volume'] + 1e-8)

    # ==========================================
    # 清除 rolling 產生的初期空值
    df = df.dropna().reset_index(drop=True)

    feature_count = len([c for c in df.columns if c not in ['date', 'target']])
    print(f"✅ 完成！共 {feature_count} 個特徵（不含 date、target）。")
    return df



def scaler(df, train_ratio=0.7):
    print("🛠️ 開始執行統一 Z-Score 縮放 (防未來洩漏機制啟動)...")
    df_scaled = df.copy()

    # 計算訓練集的截止邊界
    train_end_idx = int(len(df_scaled) * train_ratio)

    # ==========================================
    # 📌 統一對所有數值特徵進行 Z-Score（排除 date 和 target）
    # ==========================================
    feature_cols = [c for c in df_scaled.columns if c not in ['date', 'target']]

    if feature_cols:
        sc = StandardScaler()
        # 🚨 絕對關鍵：只用訓練集 (0 ~ train_end_idx) 來 fit 計算均值和標準差！
        sc.fit(df_scaled.loc[:train_end_idx, feature_cols])

        # transform 套用到全部資料 (包含測試集)
        df_scaled[feature_cols] = sc.transform(df_scaled[feature_cols])

    print(f"✅ 統一 Z-Score 完成！共縮放 {len(feature_cols)} 個特徵。")
    return df_scaled


