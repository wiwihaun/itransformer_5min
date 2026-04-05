import os
import numpy as np

def diagnose_pred_format(result_dir='./results/'):
    """
    自動尋找最新結果資料夾，並掃描 pred.npy 判斷數值是 Logits 還是已轉換的機率。
    
    參數:
        result_dir (str): 結果資料夾的根目錄路徑，預設為 './results/'
    """
    # ==========================================
    # 1. 自動尋找底下最新生成的資料夾
    # ==========================================
    if not os.path.exists(result_dir):
        print(f"🚨 找不到目錄 {result_dir}！請確認路徑是否正確。")
        return

    folders = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, f))]
    
    if not folders:
        print("🚨 找不到任何結果資料夾，請確認模型是否已跑完測試階段！")
        return
    
    # 抓取修改時間最新的一個資料夾
    latest_folder = max(folders, key=os.path.getmtime)
    file_path = os.path.join(latest_folder, 'pred.npy')
    
    print(f"📂 正在分析最新實驗結果：\n{latest_folder}\n")

    # ==========================================
    # 2. 讀取與極值分析
    # ==========================================
    try:
        # 載入 npy 檔案
        preds = np.load(file_path)

        print("✅ 檔案讀取成功！")
        print(f"➤ 原始陣列形狀 (Shape): {preds.shape}")
        print(f"➤ 總數據量: {preds.size} 個數值")

        # 掃描全部數據，找出極值
        global_min = np.min(preds)
        global_max = np.max(preds)

        print(f"\n📊 數據極值分析:")
        print(f"➤ 全局最小值 (Min): {global_min:.6f}")
        print(f"➤ 全局最大值 (Max): {global_max:.6f}")

        # ==========================================
        # 3. 終極診斷判斷
        # ==========================================
        print("\n" + "="*60)
        if global_min < 0 or global_max > 1:
            print("💡 診斷結果: 【全部數據中】確實包含負數或大於 1 的數值！")
            print("👉 結論：這是純淨的【未經轉換的 Logits】。")
            print("✅ 您可以 100% 安心使用包含內建 Sigmoid 的 StockFocalLossWithLogits (或 BCEWithLogitsLoss)！")
        else:
            print("💡 診斷結果: 【全部數據】都嚴格落在 0~1 之間。")
            print("👉 結論：框架在底層已經偷偷幫您做了 Sigmoid 轉換變成機率了。")
            print("🚨 警告：如果繼續使用 WithLogits 版本的 Loss，將會觸發 Double-Sigmoid 陷阱！請改用純 BCELoss 版本。")
        print("="*60)

    except FileNotFoundError:
        print(f"🚨 找不到檔案！請檢查資料夾內是否有 pred.npy 檔案。")
