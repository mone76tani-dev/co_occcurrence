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
- tag_communities_all_edges_louvain.csv
- startups_with_communities_louvain.csv