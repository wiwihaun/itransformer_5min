import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def features(df):
    print("🛠️ 開始計算 13 大黃金技術指標 (完全去價格化)...")
    df = df.copy()

    # ==========================================
    # 📌 1. 計算趨勢指標 (Trend)
    # ==========================================
    # MACD 與 MACD_Hist (只留 Hist 和 MACD，不留 Signal 減少雜訊)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    macd_signal = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - macd_signal

    # Bias_20 (乖離率)
    ema_20 = df['Close'].ewm(span=20, adjust=False).mean()
    df['Bias_20'] = (df['Close'] - ema_20) / ema_20 * 100

    # ==========================================
    # 📌 2. 計算動能與震盪指標 (Momentum/Oscillator)
    # ==========================================
    # RSI_14
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))

    # Stoch_K & Stoch_D (KD指標)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = (df['Close'] - low_14) / (high_14 - low_14 + 1e-8) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # CCI_20 (順勢指標)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_sma_20 = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI_20'] = (tp - tp_sma_20) / (0.015 * mad + 1e-8)

    # ==========================================
    # 📌 3. 計算波動率與爆發指標 (Volatility)
    # ==========================================
    # ATR_14 (真實波動幅度)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()

    # BB_Width & BB_PB (布林帶寬度與位置)
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    upper_band = sma_20 + (std_20 * 2)
    lower_band = sma_20 - (std_20 * 2)
    df['BB_Width'] = (upper_band - lower_band) / sma_20 * 100
    df['BB_PB'] = (df['Close'] - lower_band) / (upper_band - lower_band + 1e-8)

    # ==========================================
    # 📌 4. 計算量價資金流指標 (Volume/Money Flow)
    # ==========================================
    # OBV (能量潮)
    obv = np.where(df['Close'] > df['Close'].shift(1), df['Volume'],
          np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    df['OBV'] = pd.Series(obv).cumsum()

    # MFI_14 (資金流量指標)
    raw_mf = tp * df['Volume']
    pos_mf = np.where(tp > tp.shift(1), raw_mf, 0)
    neg_mf = np.where(tp < tp.shift(1), raw_mf, 0)
    pos_mf_sum = pd.Series(pos_mf).rolling(14).sum()
    neg_mf_sum = pd.Series(neg_mf).rolling(14).sum()
    df['MFI_14'] = 100 - (100 / (1 + (pos_mf_sum / (neg_mf_sum + 1e-8))))

    # Force_Index (強力指標)
    df['Force_Index'] = (df['Close'].diff() * df['Volume']).ewm(span=13, adjust=False).mean()

    # ==========================================
    # 1. 清除因為 rolling 產生的初期空值
    df = df.dropna().reset_index(drop=True)

    print(f"✅ 計算與清理完成！目前總特徵數: {len(df.columns) - 2} 個 (不含時間與標籤)。")
    return df



def scaler(df, train_ratio=0.7):
    print("🛠️ 開始執行分類客製化縮放 (防未來洩漏機制啟動)...")
    df_scaled = df.copy()

    # 計算訓練集的截止邊界
    # 假設您的資料是按照時間由舊到新排序好的
    train_end_idx = int(len(df_scaled) * train_ratio)

    # ==========================================
    # 📌 陣營一：無邊界特徵 (使用訓練集進行 Z-Score)
    # ==========================================
    cols_to_zscore = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Quote_asset_volume', 'Taker_buy_quote_asset_volume',
        'MACD', 'MACD_Hist', 'Bias_20', 'ATR_14',
        'BB_Width', 'OBV', 'Force_Index'
    ]

    # 確保這些欄位真的存在於 df 中
    cols_to_zscore = [c for c in cols_to_zscore if c in df_scaled.columns]

    if cols_to_zscore:
        scaler = StandardScaler()
        # 🚨 絕對關鍵：只用訓練集 (0 ~ train_end_idx) 來 fit 計算均值和標準差！
        scaler.fit(df_scaled.loc[:train_end_idx, cols_to_zscore])

        # transform 可以大膽套用到全部資料 (包含測試集)
        df_scaled[cols_to_zscore] = scaler.transform(df_scaled[cols_to_zscore])

    # ==========================================
    # 📌 陣營二：0~100 固定範圍特徵 (除以 100)
    # ==========================================
    cols_to_pct = ['RSI_14', 'Stoch_K', 'Stoch_D', 'MFI_14']
    cols_to_pct = [c for c in cols_to_pct if c in df_scaled.columns]

    if cols_to_pct:
        # 將 85 變成 0.85，完美保留邊界意義
        df_scaled[cols_to_pct] = df_scaled[cols_to_pct] / 100.0

    # ==========================================
    # 📌 陣營三：自帶基準特徵 (手動微調)
    # ==========================================
    # CCI 正常範圍約 -200 ~ 200，除以 100 讓它落在 -2 ~ 2
    if 'CCI_20' in df_scaled.columns:
        df_scaled['CCI_20'] = df_scaled['CCI_20'] / 100.0

    # BB_PB 大多在 0 ~ 1 之間，不須處理！

    return df_scaled


