import pandas as pd
import ast

PATH = "startups_with_communities_louvain.csv"
df = pd.read_csv(PATH, encoding="utf-8-sig")

# 1. コミュニティIDリストを Python のリストに変換
def parse_comm_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if s == "":
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        # 万一おかしな文字列が来たときは空リストにしておく
        return []

df["comm_list"] = df["コミュニティIDリスト"].apply(parse_comm_list)

# 2. 「少なくとも1つコミュニティに属している企業」のみカウント対象
df["num_comms"] = df["comm_list"].apply(len)
df_comm_valid = df[df["num_comms"] > 0].copy()

total_firms_with_comm = len(df_comm_valid)
print(f"少なくとも1つのコミュニティに属している企業数: {total_firms_with_comm}")

# 3. 延べカウント：企業を「コミュニティごとに」展開して数える
#    → 1社が [0,1] なら 0側・1側の両方に1社としてカウント
df_exploded = df_comm_valid[["comm_list"]].explode("comm_list")
df_exploded = df_exploded.rename(columns={"comm_list": "community_id"})

# コミュニティごとの企業数（延べ）
comm_counts = df_exploded["community_id"].value_counts().sort_index()

# 割合（母数 = total_firms_with_comm）
comm_share = (
    comm_counts
    .to_frame(name="n_firms")
    .assign(
        pct=lambda d: d["n_firms"] / total_firms_with_comm * 100
    )
)

print("\n=== コミュニティ別シェア（延べカウント） ===")
print("※母数 = 少なくとも1つコミュニティに属している企業数")
print(comm_share)

# 4. 必要なら CSV にも出力
comm_share.to_csv("community_membership_share_nobe.csv", encoding="utf-8-sig")

print("\nコミュニティ別延べシェアを出力しました → community_membership_share_nobe.csv")
