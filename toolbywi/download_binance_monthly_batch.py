import pandas as pd
import requests
import zipfile
import io
from google.colab import files

def download_binance_monthly_batch(symbol="BTCUSDT", interval="1h", start_year=2024, start_month=1, end_year=2026, end_month=2):
    print(f"🚀 開始下載 {symbol} {interval} 月包資料 ({start_year}-{start_month:02d} 到 {end_year}-{end_month:02d})...")

    # 幣安預設的 12 個欄位
    columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Close_time', 'Quote_asset_volume', 'Number_of_trades',
               'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore']

    all_dataframes = []

    # 自動產生要下載的年月清單
    periods = pd.date_range(start=f"{start_year}-{start_month}-01",
                            end=f"{end_year}-{end_month}-01",
                            freq='MS')

    for dt in periods:
        year = dt.year
        month = dt.month
        file_name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{file_name}"

        try:
            response = requests.get(url)
            if response.status_code == 404:
                print(f"⚠️ 找不到 {year}-{month:02d} 的月包 (官方可能尚未釋出)")
                continue

            response.raise_for_status()

            # 在記憶體中直接解壓縮讀取 CSV
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df_month = pd.read_csv(f, header=None, names=columns)
                    all_dataframes.append(df_month)
                    print(f"✅ 成功下載並讀取: {year}-{month:02d}")

        except Exception as e:
            print(f"❌ 下載 {year}-{month:02d} 時發生錯誤: {e}")

    if not all_dataframes:
        print("⚠️ 沒有抓到任何資料，請檢查網路。")
        return None

    # 將所有月份資料合併
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # 【關鍵修復】：強制把 Open_time 轉為數字，將混在裡面的英文字母表頭清掉
    final_df['Open_time'] = pd.to_numeric(final_df['Open_time'], errors='coerce')
    final_df = final_df.dropna(subset=['Open_time'])

    # 整理時間與轉換資料型態
    final_df['Open_time'] = pd.to_datetime(final_df['Open_time'], unit='ms')
    final_df.set_index('Open_time', inplace=True)

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume','Quote_asset_volume','Taker_buy_quote_asset_volume']
    final_df = final_df[numeric_cols].astype(float)

    print(f"\n🎉 合併完成！總共取得 {len(final_df)} 筆 K 線資料。")

    # Time-Series-Library 預設需要一個名為 'date' 的時間欄位
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'Open_time': 'date'}, inplace=True)

    return final_df