import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from scipy.special import expit
import os


def find_latest_folder(result_dir='./results/'):
    """
    自動掃描 result_dir，回傳最新生成的實驗資料夾路徑。

    參數:
        result_dir (str): 結果根目錄，預設 './results/'
    回傳:
        latest_folder (str): 最新資料夾完整路徑
    """
    if not os.path.exists(result_dir):
        raise FileNotFoundError(f"🚨 找不到目錄 {result_dir}！請確認路徑是否正確。")

    folders = [
        os.path.join(result_dir, f)
        for f in os.listdir(result_dir)
        if os.path.isdir(os.path.join(result_dir, f))
    ]

    if not folders:
        raise FileNotFoundError("🚨 找不到任何結果資料夾，請確認模型是否已跑完測試階段！")

    latest_folder = max(folders, key=os.path.getmtime)
    print(f"📂 自動偵測到最新實驗資料夾：\n{latest_folder}\n")
    return latest_folder


def load_data(csv_path, prob_threshold=0.5, result_dir='./results/',
              pred_path=None, price_csv_path=None):
    """
    載入預測結果與原始資料，對齊時間軸。
    若 csv_path 缺少 High/Low 欄位，需提供 price_csv_path（原始 OHLCV CSV）補齊。

    參數:
        csv_path       (str)  : 特徵 CSV 路徑
        prob_threshold (float): 進場機率門檻，預設 0.5
        result_dir     (str)  : 模型結果根目錄，預設 './results/'
        pred_path      (str)  : 若指定則直接使用，不自動搜尋
        price_csv_path (str)  : 原始 OHLCV CSV 路徑（含 High/Low），若特徵 CSV 已刪除則必填
    回傳:
        df_test    (DataFrame): 包含 signal / Close / High / Low 欄位的測試集
        pred_probs (ndarray)  : 預測機率
    """
    if pred_path is None:
        latest_folder = find_latest_folder(result_dir)
        pred_path = os.path.join(latest_folder, 'pred.npy')

    pred_logits  = np.load(pred_path).flatten()
    pred_probs   = expit(pred_logits)
    pred_signals = (pred_probs > prob_threshold).astype(int)
    test_len     = len(pred_probs)

    df_raw = pd.read_csv(csv_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_test = df_raw.tail(test_len).copy().reset_index(drop=True)
    df_test['signal'] = pred_signals

    # 若特徵 CSV 缺少 High/Low，從原始 OHLCV CSV 補齊
    missing_cols = [c for c in ['High', 'Low'] if c not in df_test.columns]
    if missing_cols:
        if price_csv_path is None:
            raise ValueError(
                f"特徵 CSV 缺少 {missing_cols} 欄位，請提供 price_csv_path（原始 OHLCV CSV）"
            )
        df_price = pd.read_csv(price_csv_path)
        df_price['date'] = pd.to_datetime(df_price['date'])
        df_price = df_price[['date', 'High', 'Low']]
        df_test = df_test.merge(df_price, on='date', how='left')
        if df_test[['High', 'Low']].isnull().any().any():
            raise ValueError("High/Low 合併後有 NaN，請確認兩份 CSV 的 date 欄位對齊")

    return df_test, pred_probs


def run_backtest(df_test,
                 initial_capital=1000.0,
                 position_size=0.10,
                 take_profit_pct=0.06,
                 stop_loss_pct=0.02,
                 max_hold_days=7,
                 cooldown_hours=24):
    """
    執行事件驅動回測核心邏輯。

    參數:
        df_test        (DataFrame): 含 signal / Close / High / Low / date 欄位
        initial_capital(float)    : 初始資金
        position_size  (float)    : 每次開倉比例
        take_profit_pct(float)    : 停利百分比
        stop_loss_pct  (float)    : 停損百分比
        max_hold_days  (int)      : 最大持倉天數
        cooldown_hours (int)      : 兩次進場冷卻小時數
    回傳:
        df_test     : 含 Equity 欄位的 DataFrame
        buy_records : list of (date, price)
        tp_records  : list of (date, price) — 停利
        sl_records  : list of (date, price) — 停損
        te_records  : list of (date, price) — 時間到期
    """
    capital         = initial_capital
    in_position     = False
    entry_price     = 0.0
    entry_date      = None
    invested_amount = 0.0
    last_trade_date = df_test['date'].iloc[0] - pd.Timedelta(days=10)

    equity_curve = []
    buy_records  = []
    tp_records   = []
    sl_records   = []
    te_records   = []

    for i in range(len(df_test)):
        current_date  = df_test['date'].iloc[i]
        current_close = df_test['Close'].iloc[i]
        current_high  = df_test['High'].iloc[i]
        current_low   = df_test['Low'].iloc[i]
        signal        = df_test['signal'].iloc[i]

        # 狀態 A：檢查出場
        if in_position:
            target_price = entry_price * (1 + take_profit_pct)
            stop_price   = entry_price * (1 - stop_loss_pct)

            # 停損優先（保守：假設同一根 K 線先碰低點）
            if current_low <= stop_price:
                loss = invested_amount * stop_loss_pct
                capital += (invested_amount - loss)
                sl_records.append((current_date, stop_price, entry_price))
                in_position = False

            # 停利
            elif current_high >= target_price:
                profit = invested_amount * take_profit_pct
                capital += (invested_amount + profit)
                tp_records.append((current_date, target_price, entry_price))
                in_position = False

            # 超時出場
            elif (current_date - entry_date) >= pd.Timedelta(days=max_hold_days):
                actual_return = (current_close - entry_price) / entry_price
                profit = invested_amount * actual_return
                capital += (invested_amount + profit)
                te_records.append((current_date, current_close, entry_price))
                in_position = False

        # 狀態 B：檢查進場（elif 確保出場K線不會同時進場）
        elif not in_position:
            time_since_last = current_date - last_trade_date
            if signal == 1 and time_since_last >= pd.Timedelta(hours=cooldown_hours):
                in_position     = True
                entry_price     = current_close
                entry_date      = current_date
                invested_amount = capital * position_size
                capital        -= invested_amount
                last_trade_date = current_date
                buy_records.append((current_date, current_close))

        # 結算每 K 線淨值
        if in_position:
            unrealized_value = invested_amount * (current_close / entry_price)
            total_equity = capital + unrealized_value
        else:
            total_equity = capital

        equity_curve.append(total_equity)

    df_test['Equity'] = equity_curve
    return df_test, buy_records, tp_records, sl_records, te_records


def plot_results(df_test, buy_records, tp_records, sl_records, te_records,
                 initial_capital=1000.0,
                 take_profit_pct=0.06,
                 stop_loss_pct=0.02):
    """
    繪製回測圖表：上圖為價格走勢＋買賣點位，下圖為權益曲線。
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 12),
                             gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    fig.suptitle('AI Trading Model Backtest Results (with Stop Loss)',
                 fontsize=18, fontweight='bold')

    # 上圖：價格走勢與買賣點位
    axes[0].plot(df_test['date'], df_test['Close'],
                 color='gray', alpha=0.5, label='BTC Close Price')

    if buy_records:
        buy_dates, buy_prices = zip(*buy_records)
        axes[0].scatter(buy_dates, buy_prices, marker='^', color='green',
                        s=120, label='Buy', zorder=5)

    if tp_records:
        tp_dates, tp_prices, _ = zip(*tp_records)
        axes[0].scatter(tp_dates, tp_prices, marker='*', color='gold', s=250,
                        edgecolor='black',
                        label=f'Take Profit (+{take_profit_pct*100:.0f}%)', zorder=6)

    if sl_records:
        sl_dates, sl_prices, _ = zip(*sl_records)
        axes[0].scatter(sl_dates, sl_prices, marker='X', color='magenta', s=150,
                        edgecolor='black',
                        label=f'Stop Loss (-{stop_loss_pct*100:.0f}%)', zorder=6)

    if te_records:
        te_dates, te_prices, _ = zip(*te_records)
        axes[0].scatter(te_dates, te_prices, marker='v', color='saddlebrown',
                        s=100, label='Time Exit', zorder=5)

    axes[0].set_title('Price Action & Execution Points', fontsize=14)
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='upper left', fontsize=11)

    # 下圖：權益曲線
    axes[1].plot(df_test['date'], df_test['Equity'],
                 color='blue', linewidth=2, label='Total Portfolio Value')
    axes[1].axhline(initial_capital, color='red', linestyle='--',
                    alpha=0.7, label='Initial Capital')

    peak     = df_test['Equity'].cummax()
    drawdown = (df_test['Equity'] - peak) / peak
    max_dd   = drawdown.min()

    axes[1].set_title(f'Portfolio Equity Curve (Max Drawdown: {max_dd:.2%})', fontsize=14)
    axes[1].set_ylabel('Capital ($)')
    axes[1].set_xlabel('Date')
    axes[1].legend(loc='upper left', fontsize=12)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_stats(df_test, buy_records, tp_records, sl_records, te_records,
                initial_capital=1000.0,
                take_profit_pct=0.06,
                stop_loss_pct=0.02):
    """
    輸出回測績效統計報告。
    """
    total_roi    = (df_test['Equity'].iloc[-1] - initial_capital) / initial_capital
    total_trades = len(buy_records)

    print("\n" + "="*40)
    print("💰 回測績效總結")
    print("="*40)
    print(f"🔹 最終總投資報酬率 (ROI): {total_roi:.2%}")
    print(f"🔹 總開單次數: {total_trades}")
    print("-" * 40)
    print(f"⭐ 成功停利 (+{take_profit_pct*100:.0f}%) 次數: {len(tp_records)}")
    print(f"❌ 觸發停損 (-{stop_loss_pct*100:.0f}%) 次數: {len(sl_records)}")
    print(f"⏳ 時間到期 (持有 7 天) 結算次數: {len(te_records)}")

    if total_trades > 0:
        pure_win_rate = len(tp_records) / total_trades
        print(f"🏆 純停利勝率: {pure_win_rate:.2%}")
    print("="*40)


def backtest(csv_path,
             result_dir='./results/',
             pred_path=None,
             price_csv_path=None,
             initial_capital=1000.0,
             position_size=0.10,
             take_profit_pct=0.06,
             stop_loss_pct=0.02,
             max_hold_days=7,
             cooldown_hours=24,
             prob_threshold=0.5):
    """
    一鍵執行完整回測流程：自動找資料夾 → 載入資料 → 回測 → 畫圖 → 輸出績效。

    參數:
        csv_path        (str)  : 特徵 CSV 路徑 (必填)
        result_dir      (str)  : 模型結果根目錄，自動找最新資料夾 (預設 './results/')
        pred_path       (str)  : 若指定則直接使用，不自動搜尋
        price_csv_path  (str)  : 原始 OHLCV CSV 路徑（含 High/Low），若特徵 CSV 已刪除則必填
        initial_capital (float): 初始資金 (預設 1000)
        position_size   (float): 每次開倉比例 (預設 0.10)
        take_profit_pct (float): 停利百分比 (預設 0.06)
        stop_loss_pct   (float): 停損百分比 (預設 0.02)
        max_hold_days   (int)  : 最大持倉天數 (預設 7)
        cooldown_hours  (int)  : 兩次進場冷卻小時數 (預設 24)
        prob_threshold  (float): 進場機率門檻 (預設 0.5)

    使用範例:
        from backtest import backtest
        backtest(
            csv_path='./dataset/stock/stock_features.csv',
            price_csv_path='./dataset/stock/raw_ohlcv.csv'
        )
    """
    try:
        df_test, _ = load_data(csv_path, prob_threshold, result_dir, pred_path, price_csv_path)

        df_test, buy_records, tp_records, sl_records, te_records = run_backtest(
            df_test, initial_capital, position_size,
            take_profit_pct, stop_loss_pct, max_hold_days, cooldown_hours
        )

        print("✅ 回測計算完成，正在生成圖表...")
        plot_results(df_test, buy_records, tp_records, sl_records, te_records,
                     initial_capital, take_profit_pct, stop_loss_pct)

        print_stats(df_test, buy_records, tp_records, sl_records, te_records,
                    initial_capital, take_profit_pct, stop_loss_pct)

    except Exception as e:
        print(f"🚨 發生錯誤：{e}")
