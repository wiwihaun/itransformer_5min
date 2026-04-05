import numpy as np
import pandas as pd

def target_long(df, lookahead=96, tp_pct=0.06, sl_pct=0.02):
    # 將需要運算的欄位轉換為純 NumPy 陣列以最大化效能
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)

    # 建立預設為 0 的 target 陣列
    targets = np.zeros(n, dtype=int)

    # 遍歷每一根 K 線
    for i in range(n):
        base_price = closes[i]

        # 依照當前 Close 設定止贏與止損價位
        target_up = base_price * (1 + tp_pct)
        target_down = base_price * (1 - sl_pct)

        # 確保不會超出資料邊界
        end_idx = min(n, i + 1 + lookahead)

        # 往未來 96 根檢查
        for j in range(i + 1, end_idx):
            # 保守原則：若同一根 K 線同時觸及止盈與止損，視為打止損 (0)
            if lows[j] <= target_down and highs[j] >= target_up:
                break
            # 先打止損 (標記 0 並結束此輪檢查)
            elif lows[j] <= target_down:
                break
            # 先達止贏 (標記 1 並結束此輪檢查)
            elif highs[j] >= target_up:
                targets[i] = 1
                break

    return targets

def target_short(df, lookahead=96, tp_pct=0.06, sl_pct=0.02):
    """
    計算做空 (Short) 用的目標標籤。
    - 標記 1: 優先觸及止盈價 (向下 tp_pct)
    - 標記 0: 優先觸及止損價 (向上 sl_pct) 或是兩者同時觸發(保守看待)，或是最終都未觸及
    """
    # 將需要運算的欄位轉換為純 NumPy 陣列以最大化效能
    # (已將 df_btc 修正為使用傳入的 df 變數)
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)

    # 建立預設為 0 的 target 陣列
    targets = np.zeros(n, dtype=int)

    # 遍歷每一根 K 線
    for i in range(n):
        base_price = closes[i]

        # 依照做空邏輯，當前 Close 往『下』是止贏，往『上』是止損
        target_down = base_price * (1 - tp_pct) # 止盈價 (跌下去才賺錢)
        target_up = base_price * (1 + sl_pct)   # 止損價 (漲上去就虧錢)

        # 確保不會超出資料邊界
        end_idx = min(n, i + 1 + lookahead)

        # 往未來 lookahead 根檢查
        for j in range(i + 1, end_idx):
            # 保守原則：若同一根 K 線內最高價碰到止損、最低價也碰到止盈，視為先打掉止損 (0)
            if highs[j] >= target_up and lows[j] <= target_down:
                break
            # 先打止損 (標記 0 並結束此輪檢查)
            elif highs[j] >= target_up:
                break
            # 先達止盈 (標記 1 並結束此輪檢查)
            elif lows[j] <= target_down:
                targets[i] = 1
                break

    return targets
