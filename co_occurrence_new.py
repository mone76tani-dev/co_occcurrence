# ========================================
# スタートアップ タグ共起ネットワーク解析
#  - コミュニティ検出：Louvain（全エッジ使用）
#  - 全体ネットワーク：共起100以上のみ表示（静止・ドラッグ不可）
#  - コミュニティ別ネットワーク：共起30以上のみ表示（静止・ドラッグ不可）
#  - 出力形式：PyVis HTML
# ========================================

import pandas as pd
from itertools import combinations
from collections import Counter
import networkx as nx
from pyvis.network import Network
from networkx.algorithms.community import louvain_communities

# ---------------------------
# 0. ファイルパス・パラメータ
# ---------------------------
DATA_PATH = "/Users/monetanikawa/startup_location_analysis/python_project_startup_location/非上場スタートアップ_2014以降(タグ付き).csv"

HTML_OVERALL_100 = "cooccurrence_network_overall_100plus_static.html"
HTML_COMM_PREFIX = "cooccurrence_network_community_"  # + {id}.html

# 全体ネットワーク表示用の閾値
THRESHOLD_OVERALL = 100

# コミュニティ別ネットワーク表示用の閾値（エッジ weight >= 50）
COMM_EDGE_THRESHOLD_DEFAULT = 50

# 必要なコミュニティだけ、個別に閾値を上書きしたい場合にここに書く
COMM_EDGE_THRESHOLD_BY_COMM = {
     1:30,
     2:30,
     3:5,
     4:25
}

# 小さすぎるコミュニティをスキップする場合の最小ノード数
COMM_MIN_NODES_FOR_HTML = 1  # 例: 5 にするとノード数5未満は出力しない


# ---------------------------
# 1. データ読み込み
# ---------------------------
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# ---------------------------
# 2. タグ列をリスト化
# ---------------------------
df["タグリスト"] = df["タグ"].fillna("").apply(
    lambda x: [t.strip() for t in str(x).split(",") if t.strip() != ""]
)

REMOVE_TAGS={
  # 事業形態
  "B2B","BtoB","B2C","BtoC","CtoC","D2C"}
def clean_tags(x):
    tags = [t.strip() for t in str(x).split(",") if t.strip()]
    tags = [t for t in tags if t not in REMOVE_TAGS]
    return tags

df["タグリスト"] = df["タグ"].fillna("").apply(clean_tags)


# ---------------------------
# 3. 同じ企業内のタグ組を作り、共起回数を数える
# ---------------------------
co_counts = Counter()

for tags in df["タグリスト"]:
    unique_tags = sorted(set(tags))          # 企業内の重複タグを消す
    for pair in combinations(unique_tags, 2):  # 2つ組の全てを作る
        co_counts[pair] += 1                  # 共起回数をカウント

# DataFrameへ変換（全エッジ）
edges = pd.DataFrame(
    [(a, b, w) for (a, b), w in co_counts.items()],
    columns=["tag1", "tag2", "weight"]
)

print("▼共起回数 上位10件")
print(edges.sort_values("weight", ascending=False).head(10))
print(f"\n全エッジ数: {len(edges)}")

# ---------------------------
# 4. NetworkXで「全エッジ」のグラフ構築
# ---------------------------
G_all = nx.Graph()
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
    subG = G_all.subgraph(comm)
    top_tags = sorted(subG.degree(), key=lambda x: x[1], reverse=True)[:10]

    summary_rows.append({
        "community_id": i,
        "num_tags": len(comm),
        "num_edges": subG.number_of_edges(),
        "top_tags": ", ".join([t for t, d in top_tags])
    })

    for tag in comm:
        tag_to_comm[tag] = i

summary_df = pd.DataFrame(summary_rows)
print("\n▼コミュニティ概要（上位タグ）")
print(summary_df)

summary_df.to_csv("community_summary_louvain.csv", index=False, encoding="utf-8-sig")

# タグ→コミュニティ 対応CSV
tag_comm_df = pd.DataFrame(
    [{"tag": tag, "community_id": comm_id} for tag, comm_id in tag_to_comm.items()]
)
tag_comm_df.to_csv("tag_communities_all_edges_louvain.csv", index=False, encoding="utf-8-sig")

# ---------------------------
# 6. 全体ネットワーク（共起100以上）HTML可視化（静止・ドラッグ不可）
# ---------------------------

edges_100 = edges[edges["weight"] >= THRESHOLD_OVERALL].copy()
print(f"\n閾値 {THRESHOLD_OVERALL}以上のエッジ数（可視化対象）: {len(edges_100)}")

# 100以上のエッジだけでグラフを作成（レイアウト計算用）
G_100 = nx.Graph()
for _, row in edges_100.iterrows():
    G_100.add_edge(row["tag1"], row["tag2"], weight=row["weight"])

# レイアウト計算（重なりをある程度減らすために kamada_kawai_layout を使用）
if G_100.number_of_nodes() > 0:
    pos_100 = nx.kamada_kawai_layout(G_100)
else:
    pos_100 = {}

net_overall = Network(
    height="900px",
    width="100%",
    bgcolor="#ffffff",
    font_color="#000000",
    notebook=False,
    directed=False
)

# 物理エンジンOFF＋ノードドラッグ禁止
net_overall.set_options("""
{
  "physics": { "enabled": false },
  "interaction": { "dragNodes": false },
  "layout": { "improvedLayout": false }
}
""")

# ノード追加（座標固定・コミュニティ色分け・ドラッグ不可）
for node, (x, y) in pos_100.items():
    comm_id = tag_to_comm.get(node, -1)
    net_overall.add_node(
        node,
        label=node,
        group=comm_id,
        title=f"Tag: {node}<br>Community: {comm_id}",
        x=float(x) * 1000,
        y=float(y) * 1000,
        physics=False,
        fixed=True           # ← ドラッグしても動かない
    )

# エッジ追加
for _, row in edges_100.iterrows():
    net_overall.add_edge(
        row["tag1"],
        row["tag2"],
        value=row["weight"],
        title=f"共起回数: {row['weight']}"
    )

net_overall.write_html(HTML_OVERALL_100, open_browser=False)
print(f"\n全体ネットワーク HTML 出力: {HTML_OVERALL_100}")

# ---------------------------
# 7. コミュニティ別ネットワーク（コミュニティごとの閾値で表示）HTML出力（静止・ドラッグ不可）
# ---------------------------

for i, comm in enumerate(communities):
    if len(comm) < COMM_MIN_NODES_FOR_HTML:
        continue

    comm_nodes = set(comm)

    # 🔸このコミュニティ i に対して使う閾値を決める
    #   辞書にあればその値、なければデフォルト（30）
    thr = COMM_EDGE_THRESHOLD_BY_COMM.get(i, COMM_EDGE_THRESHOLD_DEFAULT)

    # このコミュニティ内のエッジのうち、weight >= thr のものだけ
    edges_comm = edges[
        (edges["tag1"].isin(comm_nodes)) &
        (edges["tag2"].isin(comm_nodes)) &
        (edges["weight"] >= thr)
    ]

    if edges_comm.empty:
        print(f"コミュニティ {i}: weight >= {thr} のエッジなし → スキップ")
        continue

    print(f"コミュニティ {i}: 閾値={thr}, ノード数={len(comm_nodes)}, エッジ数={len(edges_comm)}")

# このコミュニティのエッジ一覧を CSV 出力
    edges_comm_out = edges_comm.copy()
    edges_comm_out["community_id"] = i   # どのコミュニティか分かるように列を追加（任意）

    csv_path = f"community_{i}_edges_thr{thr}.csv"
    edges_comm_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  → コミュニティ {i} エッジ一覧 CSV 出力: {csv_path}")

    print(f"コミュニティ {i}: 閾値={thr}, ノード数={len(comm_nodes)}, エッジ数={len(edges_comm)}")

    edges_comm_all = edges[
        (edges["tag1"].isin(comm_nodes)) &
        (edges["tag2"].isin(comm_nodes))
    ]

    edges_comm_all_out = edges_comm_all.copy()
    edges_comm_all_out["community_id"] = i
    edges_comm_all_out.to_csv(f"community_{i}_edges_all.csv", index=False, encoding="utf-8-sig")



    # グラフ構築
    G_comm = nx.Graph()
    for _, row in edges_comm.iterrows():
        G_comm.add_edge(row["tag1"], row["tag2"], weight=row["weight"])

    # レイアウト計算（ここでは weight 無視にしたいなら weight=None にしてもOK）
    pos_comm = nx.spring_layout(G_comm, seed=0, k=0.3, iterations=80)

    net_comm = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        notebook=False,
        directed=False
    )

    net_comm.set_options("""
    {
      "physics": { "enabled": false },
      "interaction": { "dragNodes": false },
      "layout": { "improvedLayout": false }
    }
    """)

    # ノード追加（座標固定・ドラッグ不可）
    for node, (x, y) in pos_comm.items():
        net_comm.add_node(
            node,
            label=node,
            group=i,
            title=f"Tag: {node}<br>Community: {i}<br>threshold: {thr}",
            x=float(x) * 1000,
            y=float(y) * 1000,
            physics=False,
            fixed=True
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
# 8. 各企業にコミュニティIDをふる
# ---------------------------
def get_comms(tag_list):
    # その企業のタグのうち、コミュニティに属しているもののID集合
    return sorted({tag_to_comm[t] for t in tag_list if t in tag_to_comm})

df["コミュニティIDリスト"] = df["タグリスト"].apply(get_comms)
df["コミュニティIDリスト_str"] = df["コミュニティIDリスト"].apply(
    lambda li: ",".join(str(x) for x in li)
)

df.to_csv("startups_with_communities_louvain.csv", index=False, encoding="utf-8-sig")

print("\n=== 完了!! ===")
print("・タグ×コミュニティ → tag_communities_all_edges_louvain.csv")
print("・企業×コミュニティ → startups_with_communities_louvain.csv")
print("・コミュニティ概要 → community_summary_louvain.csv")
print(f"・全体ネットワーク(共起>= {THRESHOLD_OVERALL}) → {HTML_OVERALL_100}")
print(f"・コミュニティ別ネットワーク → {HTML_COMM_PREFIX}{{community_id}}.html")
print("\n--- コミュニティ別ネットワーク閾値一覧 ---")
for i, comm in enumerate(communities):
    thr = COMM_EDGE_THRESHOLD_BY_COMM.get(i, COMM_EDGE_THRESHOLD_DEFAULT)
    print(f"  Community {i}: threshold = {thr} → {HTML_COMM_PREFIX}{i}.html")
