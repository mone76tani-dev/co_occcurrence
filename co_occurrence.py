# ========================================
# スタートアップ タグ共起ネットワーク解析（静止・HTML出力）
#  - コミュニティ検出：全エッジ使用（Louvain）
#  - 全体ネットワーク：共起100以上のみ可視化
#  - コミュニティ別ネットワーク：閾値なしで可視化
#  - すべて PyVis の HTML 出力 & 物理シミュレーションOFF
# ========================================

import pandas as pd
from itertools import combinations
from collections import Counter
import networkx as nx
from pyvis.network import Network
from networkx.algorithms.community import louvain_communities
import os

# ---------------------------
# 0. ファイルパス
# ---------------------------
DATA_PATH = "/Users/monetanikawa/startup_location_analysis/python_project_startup_location/非上場スタートアップ_2014以降(タグ付き).csv"

# 出力ディレクトリ
OUTPUT_DIR = "co_occurrence_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 出力ファイル名
HTML_OVERALL_100 = os.path.join(
    OUTPUT_DIR, "cooccurrence_network_overall_100plus.html"
)
HTML_COMM_PREFIX = os.path.join(
    OUTPUT_DIR, "cooccurrence_network_community_"  # + {id}.html
)

# ---------------------------
# 1. データ読み込み
# ---------------------------
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

#pd:pandas dataframeのこと

# ---------------------------
# 2. タグ列をリスト化
# ---------------------------
df["タグリスト"] = df["タグ"].fillna("").apply(
    lambda x: [t.strip() for t in str(x).split(",") if t.strip() != ""]
)

# ---------------------------
# 3. 同じ企業内のタグ組を作り、共起回数を数える（全エッジ）
# ---------------------------
co_counts = Counter()

for tags in df["タグリスト"]:
    unique_tags = sorted(set(tags))          # 企業内の重複タグを消す
    for pair in combinations(unique_tags, 2): # 2つ組の全てを作る
        co_counts[pair] += 1                  # 共起回数をカウント　co_countsはキー： (tag1, tag2) のタプル、値：共起回数という構造になっている。

# DataFrameへ変換（全エッジ）
edges = pd.DataFrame(
    [(a, b, w) for (a, b), w in co_counts.items()],
    columns=["tag1", "tag2", "weight"] #タグa、タグbと、weightって感じ
)

print("▼共起回数 上位10件")
print(edges.sort_values("weight", ascending=False).head(10)) #ascending:昇順　　#上から10行だけプリント
print(f"\n全エッジ数（weight >= 1）: {len(edges)}")

# ---------------------------
# 4. NetworkXで「全エッジ」のグラフ構築（コミュニティ検出用）
# ---------------------------
G_all = nx.Graph() #NetworkX の 無向グラフオブジェクト を1個作っている
for _, row in edges.iterrows():
    G_all.add_edge(row["tag1"], row["tag2"], weight=row["weight"])

print(f"全体グラフ ノード数: {G_all.number_of_nodes()}")
print(f"全体グラフ エッジ数: {G_all.number_of_edges()}")

# ---------------------------
# 5. Louvain法でコミュニティ検出（全エッジ使用）
# ---------------------------
communities = list(
    louvain_communities(G_all, weight="weight", resolution=1.0, seed=0)
)

print(f"\n見つかったコミュニティ数: {len(communities)}")

summary_rows = []
tag_to_comm = {}

for i, comm in enumerate(communities):
    subG = G_all.subgraph(comm) #サブフラフを作成
    top_tags = sorted(subG.degree(), key=lambda x: x[1], reverse=True)[:10]

    summary_rows.append({
        "community_id": i,
        "num_tags": len(comm),
        "num_edges": subG.number_of_edges(),
        "top_tags": ", ".join([t for t, d in top_tags])
    })

    # tag→community の対応付け
    for tag in comm:
        tag_to_comm[tag] = i

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "community_summary_louvain.csv"),
    index=False,
    encoding="utf-8-sig"
)
print("\n▼コミュニティ概要（上位タグ）")
print(summary_df)

# ---------------------------
# 6. タグ→コミュニティ 対応CSV
# ---------------------------
tag_comm_df = pd.DataFrame(
    [{"tag": tag, "community_id": comm_id} for tag, comm_id in tag_to_comm.items()]
)
tag_comm_df.to_csv(
    os.path.join(OUTPUT_DIR, "tag_communities_all_edges_louvain.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------
# 7. 全体ネットワーク（共起100以上のみ）の HTML 可視化（静止）
# ---------------------------

THRESHOLD_OVERALL = 100
edges_100 = edges[edges["weight"] >= THRESHOLD_OVERALL].copy()
print(f"\n閾値 {THRESHOLD_OVERALL}以上のエッジ数（可視化対象）: {len(edges_100)}")

# 100以上のエッジだけでグラフを作成（レイアウト用）
G_100 = nx.Graph()
for _, row in edges_100.iterrows():
    G_100.add_edge(row["tag1"], row["tag2"], weight=row["weight"])

# spring_layout でレイアウト計算（静止）
pos_100 = nx.spring_layout(G_100, seed=0, k=0.3, iterations=80)

# PyVis ネットワーク（物理エンジン OFF）
net_overall = Network(
    height="800px",
    width="100%",
    bgcolor="#ffffff",
    font_color="#000000",
    notebook=False,
    directed=False
)

# physics を完全に停止
net_overall.set_options("""
{
  "physics": {
    "enabled": false
  }
}
""")


# ノード追加（座標固定・コミュニティで色分け）
for node, (x, y) in pos_100.items():
    comm_id = tag_to_comm.get(node, -1)
    net_overall.add_node(
        node,
        label=node,
        x=float(x) * 1000,   # PyVis 用にスケール
        y=float(y) * 1000,
        physics=False,       # ノードごとの物理もOFF
        group=comm_id,
        title=f"Tag: {node}<br>Community: {comm_id}"
    )

# エッジ追加
for _, row in edges_100.iterrows():
    net_overall.add_edge(
        row["tag1"],
        row["tag2"],
        value=row["weight"],  # weight に応じて太さ
        title=f"共起回数: {row['weight']}"
    )

# write_html でテンプレートバグ回避 & ブラウザ自動起動なし
net_overall.write_html(HTML_OVERALL_100, open_browser=False)
print(f"\n全体ネットワーク HTML 出力: {HTML_OVERALL_100}")

# ---------------------------
# 8. コミュニティ別ネットワーク（閾値なし）HTML出力（静止）
# ---------------------------

# 小さすぎるコミュニティをスキップしたい場合はここを変える
COMM_MIN_NODES_FOR_HTML = 1  # 例: 5 にするとノード数5未満は出力しない

for i, comm in enumerate(communities):
    comm_nodes = set(comm)
    if len(comm_nodes) < COMM_MIN_NODES_FOR_HTML:
        continue

    # このコミュニティ内のエッジ（両端ノードがコミュニティ内にあるもの全部）
    edges_comm = edges[
        edges["tag1"].isin(comm_nodes) & edges["tag2"].isin(comm_nodes)
    ]

    print(f"コミュニティ {i}: ノード数={len(comm_nodes)}, エッジ数={len(edges_comm)}")

    # グラフ構築
    G_comm = nx.Graph()
    for _, row in edges_comm.iterrows():
        G_comm.add_edge(row["tag1"], row["tag2"], weight=row["weight"])

    # エッジが一切ない場合は、ノードだけのグラフを作る
    if G_comm.number_of_nodes() == 0:
        for n in comm_nodes:
            G_comm.add_node(n)

    # レイアウト計算
    pos_comm = nx.spring_layout(G_comm, seed=0, k=0.3, iterations=80)

    net_comm = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        notebook=False,
        directed=False
    )

    net_comm.set_options("""
    var options = {
      physics: { enabled: false }
    }
    """)

    # ノード追加
    for node, (x, y) in pos_comm.items():
        net_comm.add_node(
            node,
            label=node,
            x=float(x) * 1000,
            y=float(y) * 1000,
            physics=False,
            group=i,
            title=f"Tag: {node}<br>Community: {i}"
        )

    # エッジ追加
    for _, row in edges_comm.iterrows():
        net_comm.add_edge(
            row["tag1"],
            row["tag2"],
            value=row["weight"],
            title=f"共起回数: {row['weight']}"
        )

    html_path = f"{HTML_COMM_PREFIX}{i}.html"
    net_comm.write_html(html_path, open_browser=False)
    print(f"  → コミュニティ {i} ネットワーク HTML 出力: {html_path}")

# ---------------------------
# 9. 各企業にコミュニティIDをふる
# ---------------------------
def get_comms(tag_list):
    # その企業のタグのうち、コミュニティに属しているもののID集合
    return sorted({tag_to_comm[t] for t in tag_list if t in tag_to_comm})

df["コミュニティIDリスト"] = df["タグリスト"].apply(get_comms)
df["コミュニティIDリスト_str"] = df["コミュニティIDリスト"].apply(
    lambda li: ",".join(str(x) for x in li)
)

df.to_csv(
    os.path.join(OUTPUT_DIR, "startups_with_communities_louvain.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n=== 完了!! ===")
print(f"・タグ×コミュニティ → {os.path.join(OUTPUT_DIR, 'tag_communities_all_edges_louvain.csv')}")
print(f"・企業×コミュニティ → {os.path.join(OUTPUT_DIR, 'startups_with_communities_louvain.csv')}")
print(f"・コミュニティ概要 → {os.path.join(OUTPUT_DIR, 'community_summary_louvain.csv')}")
print(f"・全体ネットワーク(共起>= {THRESHOLD_OVERALL}) → {HTML_OVERALL_100}")
print(f"・コミュニティ別ネットワーク → {HTML_COMM_PREFIX}{{community_id}}.html")
