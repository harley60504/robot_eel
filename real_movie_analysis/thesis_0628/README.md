# Thesis real-swimming figure package after 2026-06-28

這包資料是從 `real_movie_analysis` 中 2026-06-29、2026-07-01、2026-07-02、2026-07-03 的分析輸出整理出來，只做複製整理，沒有刪除或修改原始分析資料。

## Folder guide

- `01_real`：實際影片截出的姿態圖，包含 5 張代表圖的 contact sheet、clean frames、annotated frames，以及 7-dot 檢查圖。
- `02_drawn`：由 LED 位置重畫出的姿態示意圖，包含 relative pose/adaptive/teal-red 類型圖。
- `03_9grid`：最後用於論文比較的 9 宮格軌跡圖、單張 panel、以及代表圖 manifest。
- `03_9grid/selected`：最推薦直接放論文的 9 宮格圖與來源 manifest。
- `04_tables`：四日期所有正式 trial 的平均表與逐筆 trial 數據。
- `04_tables/selected`：推薦寫結果段落時引用的統計表副本。
- `00_notes/manifest.csv`：每個整理檔案的來源路徑與複製後位置。
- `00_notes/category_counts.csv`：各分類檔案數。

## Recommended thesis files

論文主圖建議優先使用：
`03_9grid/selected/final_9grid_time_colored_per_lap.png`

數據表建議優先引用：
`04_tables/selected/condition_summary.csv`

若要補充姿態變化，先從 `01_real` 找真實截圖，再從 `02_drawn` 找畫出的身形示意圖。每張圖的原始來源都可以從 `00_notes/manifest.csv` 回查。
