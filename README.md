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

