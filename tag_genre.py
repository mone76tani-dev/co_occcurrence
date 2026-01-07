import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# 0. 設定
# =========================================================
DATA_PATH = '/Users/monetanikawa/startup_location_analysis/python_project_startup_location/非上場スタートアップ_2014以降(タグ・事業内容・評価額付き).csv'

TAG_COL = "タグ"        # タグ列名
DESC_COL = "事業内容"   # 事業内容テキスト列名
LOC_COL = "LocName"    # 町丁目情報（例：東京都/渋谷区/渋谷/２丁目）

OUT_TAGMAP   = "tag2category_okamoto7.csv"
OUT_STARTUP  = "startups_with_categories_tags_and_text.csv"

OUT_PRIMARY_TAGS    = "chome_category_primary_tags.csv"
OUT_MULTI_TAGS      = "chome_category_multilabel_tags.csv"
OUT_FRACTION_TAGS   = "chome_category_fractional_tags.csv"
OUT_PRIMARY_TEXT    = "chome_category_primary_text.csv"

# =========================================================
# 1. データ読み込み & タグをリスト化
# =========================================================
df = pd.read_csv(DATA_PATH)

df["tag_list"] = (
    df[TAG_COL]
    .fillna("")
    .astype(str)
    .apply(lambda s: [t.strip() for t in s.split(",") if t.strip() != ""])
)

print("rows:", len(df))
print("example tags:", df["tag_list"].iloc[0])

# =========================================================
# 2. 岡本さん 7分類の「アンカー」定義（タグ用）
#    ★ここは好きに増やしてOK
# =========================================================
anchors = {
    "メディア・エンタメ": [
        "メディア","エンタメ","教育","学習","動画","ゲーム",
        "SNS","コミュニティ","ライフイベント","VR","コンテンツ"
    ],
    "医療・ヘルスケア": [
        "医療","ヘルスケア","健康管理","医療機器","介護",
        "製薬","バイオ","再生医療","バイオテクノロジー","MedTech"
    ],
    "IT・コンサルティング": [
        "IT","コンサルティング","SaaS","AI","人工知能",
        "データ分析","機械学習","ディープラーニング",
        "システム開発","ソフトウェア","クラウドサービス",
        "業務効率化","AdTech","MarTech","情報サービス","プラットフォーム"
    ],
    "小売・EC": [
        "小売","EC","eコマース","通販","物流","フード","食",
        "食品・飲料","シェアリング","モビリティ","自動車",
        "ドローン","リテール"
    ],
    "金融・決済": [
        "金融","FinTech","決済","ブロックチェーン",
        "資産運用","資産管理","会計","家計管理","仮想通貨",
        "融資","レンディング","決済代行"
    ],
    "レジャー・不動産": [
        "レジャー","不動産","不動産管理","不動産売買",
        "賃貸","旅行","スポーツ","予約","建設","インフラ"
    ],
    "HR・採用": [
        "HR","採用","採用支援","人材","人材育成","人事制度",
        "労務","転職","副業","クラウドソーシング","バックオフィス支援"
    ],
}

# ルール分類用逆引き辞書
rule_map = {}
for cat, words in anchors.items():
    for w in words:
        rule_map[w] = cat

# =========================================================
# 3. 全タグ一覧 → ルール＋BERTでカテゴリ付与
# =========================================================
all_tags = sorted({t for tags in df["tag_list"] for t in tags})
print("unique tags:", len(all_tags))

tag2cat = {}
tag2how = {}

# まずルール
for t in all_tags:
    if t in rule_map:
        tag2cat[t] = rule_map[t]
        tag2how[t] = "rule"
    else:
        tag2cat[t] = None
        tag2how[t] = None

unmapped_tags = [t for t in all_tags if tag2cat[t] is None]
print("unmapped after rule:", len(unmapped_tags))

# BERTモデル読み込み（タグ用＆テキスト用で共通に使う）
print("Loading Japanese Sentence-BERT model...")
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

# カテゴリごとのアンカー単語をembedding → 平均ベクトル
anchor_vecs = {}
for cat, words in anchors.items():
    vecs = model.encode(words, normalize_embeddings=True)
    anchor_vecs[cat] = vecs.mean(axis=0, keepdims=True)

def bert_assign_tag(tag, threshold=0.35):
    v = model.encode([tag], normalize_embeddings=True)  # (1, dim)
    sims = {cat: cosine_similarity(v, anchor_vecs[cat])[0, 0] for cat in anchors}
    best_cat = max(sims, key=sims.get)
    best_sim = sims[best_cat]
    if best_sim >= threshold:
        return best_cat, best_sim
    else:
        return None, best_sim

bert_threshold = 0.35  # 必要なら 0.3〜0.5 で調整

print("Assigning categories to remaining tags by BERT...")
for t in tqdm(unmapped_tags):
    cat, sim = bert_assign_tag(t, threshold=bert_threshold)
    tag2cat[t] = cat
    if cat is not None:
        tag2how[t] = f"bert({sim:.2f})"
    else:
        tag2how[t] = f"unclassified({sim:.2f})"

# タグ→カテゴリ表を書き出し（後で目視チェック用）
tagmap_df = pd.DataFrame({
    "tag": all_tags,
    "category": [tag2cat[t] for t in all_tags],
    "assigned_by": [tag2how[t] for t in all_tags],
})
tagmap_df.to_csv(OUT_TAGMAP, index=False, encoding="utf-8-sig")
print("saved tag map:", OUT_TAGMAP)

# =========================================================
# 4. 企業ごとにカテゴリ付与（タグベース）
# =========================================================
def tags_to_categories(tags):
    cats = [tag2cat.get(t) for t in tags]
    return sorted({c for c in cats if c is not None})

def primary_from_tags(tags):
    cats = [tag2cat.get(t) for t in tags]
    cats = [c for c in cats if c is not None]
    if len(cats) == 0:
        return None
    return Counter(cats).most_common(1)[0][0]

df["categories_tags"]    = df["tag_list"].apply(tags_to_categories)
df["primary_from_tags"]  = df["tag_list"].apply(primary_from_tags)

print("sample categories_tags:", df["categories_tags"].head(3))

# =========================================================
# 5. 事業内容テキストから primary を決める（キーワード＋BERT）
# =========================================================

# --- 5-1. カテゴリごとのキーワード辞書 ---
# （タグアンカーと似ているが、文章中に出そうな単語を意識）
category_keywords = {
    "メディア・エンタメ": [
        "メディア","エンタメ","マンガ","アニメ","ゲーム","動画",
        "番組","配信","ライブ","イベント","コミュニティ"
    ],
    "医療・ヘルスケア": [
        "医療","ヘルスケア","病院","クリニック","患者","診療",
        "在宅医療","看護","介護","薬","健康","予防","検診"
    ],
    "IT・コンサルティング": [
        "AI","人工知能","SaaS","システム","ソフトウェア","アプリ",
        "プラットフォーム","IT","DX","デジタル","コンサルティング",
        "業務効率化","データ分析","クラウド"
    ],
    "小売・EC": [
        "EC","通販","オンラインストア","小売","店舗","決済端末",
        "販売","商品","カート","ショッピング","出店"
    ],
    "金融・決済": [
        "金融","FinTech","決済","口座","送金","融資","ローン",
        "資産運用","資産管理","証券","保険","与信","請求"
    ],
    "レジャー・不動産": [
        "不動産","賃貸","売買","物件","マンション","オフィス",
        "レジャー","旅行","観光","ホテル","宿泊","キャンプ"
    ],
    "HR・採用": [
        "採用","人材","人事","労務","転職","求人","評価制度",
        "タレントマネジメント","副業","リスキリング","研修"
    ],
}

def classify_by_keywords(text):
    text = str(text)
    scores = {cat: 0 for cat in category_keywords}
    for cat, kws in category_keywords.items():
        for kw in kws:
            scores[cat] += len(re.findall(re.escape(kw), text))
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        return None, scores
    return best_cat, scores

# --- 5-2. カテゴリの「代表文」を作り embedding ---
category_labels = {
    "メディア・エンタメ": "メディアやエンターテインメント、ゲームや動画配信などのサービス",
    "医療・ヘルスケア": "病院やクリニック、在宅医療、ヘルスケア、健康管理に関するサービス",
    "IT・コンサルティング": "AIやSaaSなどのITサービス、DX推進、業務効率化やコンサルティング",
    "小売・EC": "ECサイトやオンラインストア、小売業、商品販売に関するサービス",
    "金融・決済": "金融サービス、決済、送金、融資、資産運用などに関するサービス",
    "レジャー・不動産": "不動産や賃貸、旅行、観光、宿泊、レジャーに関するサービス",
    "HR・採用": "採用支援、人材マネジメント、労務管理、研修や人事制度に関するサービス",
}

cat_texts = [category_labels[c] for c in anchors.keys()]
cat_names  = list(anchors.keys())
cat_embs   = model.encode(cat_texts, normalize_embeddings=True)

def classify_by_bert_text(text):
    v = model.encode([str(text)], normalize_embeddings=True)
    sims = cosine_similarity(v, cat_embs)[0]  # 各カテゴリとの類似度
    idx = int(np.argmax(sims))
    return cat_names[idx], float(sims[idx])

# --- 5-3. ハイブリッド判定 ---
def decide_primary_from_text(text, kw_first=True, bert_threshold=0.35):
    """
    1. キーワードでスコア > 0 のカテゴリがあればそれを優先
       （同点が複数あればBERTでタイブレーク）
    2. 全カテゴリ0点なら BERT だけで判定（類似度閾値付き）
    """
    if not isinstance(text, str) or text.strip() == "":
        return None, "empty"

    # キーワード
    cat_kw, scores_kw = classify_by_keywords(text)

    if cat_kw is not None:
        # 同点カテゴリを拾う
        max_score = max(scores_kw.values())
        best_cats = [c for c, s in scores_kw.items() if s == max_score]
        if len(best_cats) == 1:
            return best_cats[0], f"keyword(max={max_score})"
        else:
            # 複数同点ならBERTで一番近いカテゴリに
            _, sims = [], []
            v = model.encode([text], normalize_embeddings=True)
            # best_catsの中で最も類似度が高いカテゴリを選ぶ
            best_cat = None
            best_sim = -1
            for c in best_cats:
                idx = cat_names.index(c)
                sim = cosine_similarity(v, cat_embs[idx:idx+1])[0, 0]
                if sim > best_sim:
                    best_sim = sim
                    best_cat = c
            return best_cat, f"keyword+tiebreak_bert(sim={best_sim:.2f})"

    # キーワードが全滅 → BERTのみ
    cat_b, sim_b = classify_by_bert_text(text)
    if sim_b >= bert_threshold:
        return cat_b, f"bert_only(sim={sim_b:.2f})"
    else:
        return None, f"bert_low(sim={sim_b:.2f})"

# --- 5-4. 実行 ---
primary_text_list = []
text_method_list  = []

print("Classifying primary_from_text by keyword + BERT...")
for txt in tqdm(df[DESC_COL].fillna("").astype(str)):
    cat, how = decide_primary_from_text(txt)
    primary_text_list.append(cat)
    text_method_list.append(how)

df["primary_from_text"] = primary_text_list
df["text_method"]       = text_method_list

print(df[[DESC_COL, "primary_from_tags", "primary_from_text", "text_method"]].head(5))

# =========================================================
# 6. タグ由来＋テキスト由来の和集合カテゴリも持たせる
# =========================================================
def union_categories(row):
    cats = set(row.get("categories_tags", []))
    if row.get("primary_from_text") is not None:
        cats.add(row["primary_from_text"])
    return sorted(cats)

df["all_categories_union"] = df.apply(union_categories, axis=1)

# 保存
df.to_csv(OUT_STARTUP, index=False, encoding="utf-8-sig")
print("saved enriched startup data:", OUT_STARTUP)

# =========================================================
# 7. 東京都だけ抜き出して町丁目 × 分野で集計
# =========================================================
df_tokyo = df[df[LOC_COL].astype(str).str.startswith("東京都/")].copy()

# --- 7-1. primary_from_tags でカウント ---
primary_tags_df = (
    df_tokyo
    .dropna(subset=["primary_from_tags"])
    .groupby([LOC_COL, "primary_from_tags"])
    .size()
    .reset_index(name="count")
)
primary_tags_df.to_csv(OUT_PRIMARY_TAGS, index=False, encoding="utf-8-sig")
print("saved:", OUT_PRIMARY_TAGS)

# --- 7-2. categories_tags（マルチラベルで重複カウント） ---
rows = []
for _, r in df_tokyo.iterrows():
    for cat in r["categories_tags"]:
        rows.append([r[LOC_COL], cat])

multi_df = pd.DataFrame(rows, columns=[LOC_COL, "category"])
multi_agg = (
    multi_df
    .groupby([LOC_COL, "category"])
    .size()
    .reset_index(name="count")
)
multi_agg.to_csv(OUT_MULTI_TAGS, index=False, encoding="utf-8-sig")
print("saved:", OUT_MULTI_TAGS)

# --- 7-3. categories_tags を按分カウント ---
rows = []
for _, r in df_tokyo.iterrows():
    cats = r["categories_tags"]
    if len(cats) == 0:
        continue
    w = 1.0 / len(cats)
    for cat in cats:
        rows.append([r[LOC_COL], cat, w])

frac_df = pd.DataFrame(rows, columns=[LOC_COL, "category", "weight"])
frac_agg = (
    frac_df
    .groupby([LOC_COL, "category"])["weight"]
    .sum()
    .reset_index()
)
frac_agg.to_csv(OUT_FRACTION_TAGS, index=False, encoding="utf-8-sig")
print("saved:", OUT_FRACTION_TAGS)

# --- 7-4. primary_from_text でカウント（テキスト版） ---
primary_text_df = (
    df_tokyo
    .dropna(subset=["primary_from_text"])
    .groupby([LOC_COL, "primary_from_text"])
    .size()
    .reset_index(name="count")
)
primary_text_df.to_csv(OUT_PRIMARY_TEXT, index=False, encoding="utf-8-sig")
print("saved:", OUT_PRIMARY_TEXT)
