import numpy as np
import pandas as pd
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


def verify_alignment(csv_path, result_dir='./results/', pred_path=None, true_path=None):
    """
    驗證 CSV 的 target 欄位與模型輸出的 true.npy 是否對齊。
    若未指定 pred_path / true_path，自動從 result_dir 找最新資料夾讀取。

    參數:
        csv_path   (str): 特徵 CSV 路徑 (必填)
        result_dir (str): 模型結果根目錄，預設 './results/'
        pred_path  (str): 指定 pred.npy 路徑（不指定則自動搜尋）
        true_path  (str): 指定 true.npy 路徑（不指定則自動搜尋）

    回傳:
        verify_df (DataFrame): 含對齊結果的完整比對表
    """
    try:
        # 自動尋找資料夾
        if pred_path is None or true_path is None:
            latest_folder = find_latest_folder(result_dir)
            if pred_path is None:
                pred_path = os.path.join(latest_folder, 'pred.npy')
            if true_path is None:
                true_path = os.path.join(latest_folder, 'true.npy')

        # 載入預測與真實陣列
        pred_logits  = np.load(pred_path).flatten()
        true_labels  = np.load(true_path).flatten()
        pred_probs   = expit(pred_logits)
        test_len     = len(pred_probs)

        # 載入 CSV
        df_raw  = pd.read_csv(csv_path)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_test = df_raw.tail(test_len).copy().reset_index(drop=True)

        # 建立對齊比對表
        verify_df = pd.DataFrame({
            'Date'          : df_test['date'],
            'Close_Price'   : df_test['Close'].round(2),
            'CSV_Target'    : df_test['target'],
            'Model_True_NPY': true_labels,
            'Pred_Prob'     : pred_probs.round(4)
        })

        verify_df['Is_Aligned']  = verify_df['CSV_Target'] == verify_df['Model_True_NPY']
        alignment_score = verify_df['Is_Aligned'].mean()

        # 輸出檢驗報告
        print("="*50)
        print("🕰️ 時間序列對齊診斷報告")
        print("="*50)
        print(f"✅ 對齊準確率: {alignment_score:.2%}")
        if alignment_score == 1.0:
            print("🎉 完美對齊！CSV target 與 Model True NPY 完全吻合。")
        elif alignment_score >= 0.95:
            print("⚠️ 對齊率略低，建議檢查資料切分邏輯。")
        else:
            print("🚨 對齊率過低，時間軸可能錯位，請重新確認資料前處理流程！")
        print("="*50)

        return verify_df

    except Exception as e:
        print(f"🚨 發生錯誤：{e}")
        return None
