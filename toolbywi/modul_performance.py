import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit

def model_performance(result_dir='./results/', threshold=0.5, csv_path=None):
    """
    評估模型 BCE 分類結果並視覺化

    參數:
        result_dir (str): 結果資料夾的根目錄路徑
        threshold (float): 判斷為 1 (看漲) 的機率門檻值，預設為 0.5
        csv_path (str|None): 若提供，會讀取該 CSV 的 date 欄對齊 pred_signals，
            於第 3 張圖以真實日期為 x 軸；若為 None 或檔案不存在則改用樣本索引。
    """
    # ==========================================
    # 1. 自動尋找底下最新生成的資料夾
    # ==========================================
    folders = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, f))]
    if not folders:
        print("🚨 找不到任何結果資料夾，請確認模型是否已跑完測試階段！")
        return
    
    latest_folder = max(folders, key=os.path.getmtime)
    print(f"📂 正在分析最新 BCE 分類實驗結果：\n{latest_folder}")
    print(f"⚙️ 目前設定的決策門檻 (Threshold): {threshold}\n")

    # ==========================================
    # 2. 載入原始預測值與真實值
    # ==========================================
    try:
        preds_logits = np.load(os.path.join(latest_folder, 'pred.npy'))
        trues_raw = np.load(os.path.join(latest_folder, 'true.npy'))
    except FileNotFoundError:
        print("🚨 資料夾中缺少 pred.npy 或 true.npy 檔案！")
        return

    # 只取第一天 (pred_len=1) 以及 target 欄位的結果，維度調整為 (樣本數,)
    logits = preds_logits[:, 0, -1]
    true_raw = trues_raw[:, 0, -1]

    # ==========================================
    # 3. 資料處理與閾值判定
    # ==========================================
    pred_probabilities = expit(logits)
    
    # 🚨 關鍵：使用傳入的 threshold 變數作為門檻
    pred_signals = (pred_probabilities > threshold).astype(int)

    # 將標準化後的真實值還原為 0 與 1
    true_signals = (true_raw > true_raw.mean()).astype(int)

    # ==========================================
    # 4. 混淆矩陣與績效計算
    # ==========================================
    TP = ((pred_signals == 1) & (true_signals == 1)).sum() 
    TN = ((pred_signals == 0) & (true_signals == 0)).sum() 
    FP = ((pred_signals == 1) & (true_signals == 0)).sum() 
    FN = ((pred_signals == 0) & (true_signals == 1)).sum() 

    cm = np.array([[TN, FP], [FN, TP]])
    accuracy = (TP + TN) / len(true_signals)
    
    # 防呆機制：避免分母為 0 的錯誤
    precision = (TP / (TP + FP) * 100) if (TP + FP) > 0 else 0.0

    # ==========================================
    # 5. 視覺化圖表
    # ==========================================
    plt.figure(figsize=(20, 6))

    # 圖表 1：混淆矩陣熱力圖
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    plt.xticks([0.5, 1.5], ['Predicted Wrong (0)', 'Predicted Correct (1)'], fontsize=12)
    plt.yticks([0.5, 1.5], ['Actual Wrong (0)', 'Actual Correct (1)'], fontsize=12)
    plt.title(f'Binary Classification Confusion Matrix\n(Overall Accuracy: {accuracy:.2%})', fontsize=14)
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Reality', fontsize=12)

    # 圖表 2：預測機率分佈圖
    plt.subplot(1, 3, 2)
    plt.hist(pred_probabilities, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
    
    # 畫出動態的門檻值輔助線
    plt.axvline(x=threshold, color='#d62728', linestyle='--', linewidth=2, label=f'Decision Threshold ({threshold})')
    
    plt.title('Prediction Probability Distribution', fontsize=14)
    plt.xlabel('Probability of Output (Sigmoid Applied)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()

    # 圖表 3：pred_signals 時間分佈 — 檢視是否出現連續 1
    plt.subplot(1, 3, 3)
    test_len = len(pred_signals)

    if csv_path is not None and os.path.exists(csv_path):
        # 沿用 backtest.py 的對齊慣例（pred_len=1 時正確）
        df_raw = pd.read_csv(csv_path)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        x_axis = df_raw['date'].tail(test_len).reset_index(drop=True).values
        xlabel = 'Date'
        rotate = True
    else:
        x_axis = np.arange(test_len)
        xlabel = 'Sample Index (time order)'
        rotate = False
        if csv_path is not None:
            print(f"⚠️  csv_path 不存在：{csv_path}，第 3 張圖改用樣本索引。")

    plt.step(x_axis, pred_signals, where='post', color='#1f77b4', linewidth=1.0, label='pred_signal')
    ones_mask = pred_signals == 1
    plt.scatter(np.asarray(x_axis)[ones_mask], np.ones(int(ones_mask.sum())),
                s=10, color='#d62728', alpha=0.7, label='signal == 1')
    plt.yticks([0, 1])
    plt.ylim(-0.1, 1.2)
    plt.title('Pred Signal Over Time', fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('pred_signal (0/1)', fontsize=12)
    plt.legend(loc='upper right')
    if rotate:
        plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 6. 輸出文字報告
    # ==========================================
    print("="*40)
    print("🎯 測試結果統計報告")
    print("="*40)
    print(f"總測試樣本數 : {len(true_signals)} 天")
    print(f"預測看漲(1)次數: {pred_signals.sum()} 次 (模型出手頻率)")
    print(f"預測看跌(0)次數: {len(pred_signals) - pred_signals.sum()} 次")
    print("-" * 40)
    print(f"✅ 勝率 (Precision): {precision:.2f} % (出手預測 1 中，真正獲利的比例)")
    # 連續 1 (run-length) 統計：量化「是否存在連續」
    if pred_signals.sum() > 0:
        padded = np.concatenate(([0], pred_signals, [0]))
        diffs = np.diff(padded)
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]
        run_lengths = run_ends - run_starts
        max_run = int(run_lengths.max())
        num_runs = int(len(run_lengths))
        consec_runs = int((run_lengths >= 2).sum())
        print(f"📈 訊號 1 連續情形：共 {num_runs} 段，最長連續 {max_run} 期，"
              f"≥2 期連續的段數 {consec_runs}")
    else:
        print("📈 訊號 1 連續情形：本次無任何 pred_signal == 1。")
    print("="*40)