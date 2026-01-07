###########################################################################################################################################
# co_occurrence

## Overview
This project analyzes tag co-occurrence among Japanese startups
and detects tag communities using the Louvain method.


## Input
- 非上場スタートアップ_2014以降(タグ付き).csv
  - 必須列：
    - タグ（カンマ区切り）

## Run
python co_occurrence.py

## Output
- community_summary_louvain.csv  
  - 各コミュニティの概要（community_id、含まれるタグ数、主要タグなど）

- tag_communities_all_edges_louvain.csv  
  - 各タグが属するコミュニティIDの一覧（tag, community_id）

- startups_with_communities_louvain.csv  
  - 元の企業データに以下の列を追加  
    - コミュニティIDリスト  
    - コミュニティIDリスト_str（文字列形式）

- cooccurrence_network_overall_100plus_static.html  
  - 共起回数が100以上のエッジのみを用いた全体ネットワークの可視化（静止HTML）

- cooccurrence_network_community_{i}.html  
  - 各コミュニティごとのタグ共起ネットワーク可視化（i = community_id）

---

### ④ Run log / Experiment memo
ここに **「今回どう回したか」** を残す。

```md
## Run log

### 2026-01-07
- Input: 非上場スタートアップ_2014以降(タグ付き).csv
- Threshold: 100
- Purpose:
  - タグ間の強い共起構造を把握するため
- Notes:
  - 医療×AI周辺で大きなコミュニティが形成された


###########################################################################################################################################
## tag_genre.py（タグ×事業内容 → 7分類カテゴリ付与 & 町丁目集計）

### Overview
`tag_genre.py` は、スタートアップの **タグ** と **事業内容テキスト** を用いて、岡本さんの **7分類（分野）** に自動でカテゴリ付与し、さらに **東京都の町丁目（LocName）単位**で分野別に集計するスクリプトです。

### Input
- CSV: `非上場スタートアップ_2014以降(タグ・事業内容・評価額付き).csv`
- 使用列
  - `タグ`（カンマ区切り）
  - `事業内容`（文章）
  - `LocName`（例：東京都/渋谷区/渋谷/２丁目）

### Method（high level）
1. タグをリスト化（`tag_list`）
2. 7分類ごとのアンカー語（`anchors`）を定義
3. 全タグにカテゴリを付与  
   - まずルール（アンカー語に一致するもの）  
   - 残りは日本語Sentence-BERTでカテゴリ類似度を計算し、閾値（既定 `0.35`）以上で割当
4. 企業ごとにタグ由来カテゴリを付与（`categories_tags`, `primary_from_tags`）
5. 事業内容テキストから代表カテゴリを推定（キーワード → 同点はBERTでタイブレーク、全カテゴリ0点ならBERTのみ）
6. 東京都（`LocName` が `東京都/` で始まる）に絞って、町丁目×カテゴリで集計CSVを出力

### How to run
```bash
python tag_genre.py