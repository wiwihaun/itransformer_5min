import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit

def model_performance(result_dir='./results/', threshold=0.5):
    """
    評估模型 BCE 分類結果並視覺化
    
    參數:
        result_dir (str): 結果資料夾的根目錄路徑
        threshold (float): 判斷為 1 (看漲) 的機率門檻值，預設為 0.5
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
    plt.figure(figsize=(14, 6))

    # 圖表 1：混淆矩陣熱力圖
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    plt.xticks([0.5, 1.5], ['Predicted Wrong (0)', 'Predicted Correct (1)'], fontsize=12)
    plt.yticks([0.5, 1.5], ['Actual Wrong (0)', 'Actual Correct (1)'], fontsize=12)
    plt.title(f'Binary Classification Confusion Matrix\n(Overall Accuracy: {accuracy:.2%})', fontsize=14)
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Reality', fontsize=12)

    # 圖表 2：預測機率分佈圖
    plt.subplot(1, 2, 2)
    plt.hist(pred_probabilities, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
    
    # 畫出動態的門檻值輔助線
    plt.axvline(x=threshold, color='#d62728', linestyle='--', linewidth=2, label=f'Decision Threshold ({threshold})')
    
    plt.title('Prediction Probability Distribution', fontsize=14)
    plt.xlabel('Probability of Output (Sigmoid Applied)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    
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
    print("="*40)