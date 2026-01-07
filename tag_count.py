import pandas as pd
from collections import Counter

# Excelファイルの読み込み
df = pd.read_excel("/Users/monetanikawa/Downloads/作業用‼️INITIAL_三菱地所様提供データ.xlsx")  # ←ファイル名を指定
sheet= "tags"

# 対象の列名を指定
target_col = "タグ"

# 列を文字列として読み込み（NaNを除外）
words = df[target_col].dropna().astype(str)

# スペース区切りで単語に分割（必要に応じて）
all_words = []
for text in words:
    all_words.extend(text.split())

# 単語の出現回数をカウント
word_counts = Counter(all_words)

# 結果をDataFrame化
result = pd.DataFrame(word_counts.items(), columns=["単語", "出現回数"]).sort_values("出現回数", ascending=False)

# 結果の確認
print(f"総単語数: {len(all_words)}")
print(f"ユニーク単語数: {len(word_counts)}")
print(result.head(20))  # 上位20単語を表示

# 必要ならCSV出力
result.to_csv("word_count_result.csv", index=False)