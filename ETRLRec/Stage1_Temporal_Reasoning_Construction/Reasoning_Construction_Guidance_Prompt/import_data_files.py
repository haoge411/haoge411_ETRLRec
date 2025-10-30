from datasets import load_dataset
import pandas as pd
from pathlib import Path

def import_interaction(user_data_path, item_data_path, interaction_history_out_path):
    reviews = load_dataset(f"{user_data_path}") #reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Video_Games", split="full", trust_remote_code=True)
    items = load_dataset(f"{item_data_path}") #items = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Video_Games", split="full", trust_remote_code=True)


    df_r = reviews.to_pandas()[["user_id", "asin", "parent_asin", "timestamp"]].copy()
    df_i = items.to_pandas()
    keep_i = [c for c in ["asin", "parent_asin", "title"] if c in df_i.columns]
    df_i = df_i[keep_i].copy()


    asin_to_title, parent_to_title = {}, {}
    if "asin" in df_i.columns:
        tmp = df_i.dropna(subset=["asin", "title"]).drop_duplicates(subset=["asin"])
        asin_to_title = dict(zip(tmp["asin"], tmp["title"]))
    if "parent_asin" in df_i.columns:
        tmp = df_i.dropna(subset=["parent_asin", "title"]).drop_duplicates(subset=["parent_asin"])
        parent_to_title = dict(zip(tmp["parent_asin"], tmp["title"]))

    def resolve_title(row):
        return asin_to_title.get(row.get("asin")) or parent_to_title.get(row.get("parent_asin"))

    df_r["item_title"] = df_r.apply(resolve_title, axis=1)
    df_r = df_r.dropna(subset=["item_title"]).copy()


    df_r["timestamp"] = pd.to_numeric(df_r["timestamp"], errors="coerce")
    df_r = df_r.dropna(subset=["timestamp"])
    df_r = df_r.sort_values(["user_id", "timestamp"], ascending=[True, True])

    # Filter: Only retain users with the number of item titles ≥ 4.
    uniq_counts = (
        df_r.drop_duplicates(subset=["user_id", "item_title"])
            .groupby("user_id").size()
    )
    keep_uids = uniq_counts[uniq_counts >= 4].index
    df_r = df_r[df_r["user_id"].isin(keep_uids)].copy()



    uid_order = list(dict.fromkeys(df_r["user_id"]))  
    uid_map = {u: i+1 for i, u in enumerate(uid_order)}
    df_r["user_id"] = df_r["user_id"].map(uid_map)


    interaction_history_out_path = Path(interaction_history_out_path)
    df_r[["user_id", "item_title"]].to_csv(interaction_history_out_path, sep="\t", header=True, index=False)
    print(f"interaction history saved in: {interaction_history_out_path.resolve()}")
    return str(interaction_history_out_path)

def import_item(item_data_path, item_out_path):
    items = load_dataset(f"{item_data_path}") #items = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Video_Games", split="full", trust_remote_code=True)

    df_i = items.to_pandas()
    keep_i = [c for c in ["parent_asin", "asin", "title"] if c in df_i.columns]
    df_i = df_i[keep_i].copy()

    
    for col in ("parent_asin", "asin"):
        if col not in df_i.columns:
            df_i[col] = pd.NA

    
    df_i["parent_asin"] = df_i["parent_asin"].combine_first(df_i["asin"])

    
    df_i = df_i.dropna(subset=["parent_asin", "title"]).copy()
    df_i["parent_asin"] = df_i["parent_asin"].astype(str)
    df_i["title"] = df_i["title"].astype(str)
    df_i = df_i.drop_duplicates(subset=["parent_asin"], keep="first").copy()
    df_i = df_i.sort_values("parent_asin", kind="stable").reset_index(drop=True)

    
    df_i.insert(0, "id", range(1, len(df_i) + 1))

    item_out_path = Path(item_out_path)
    item_out_path.parent.mkdir(parents=True, exist_ok=True)
    df_i[["id", "parent_asin", "title"]].to_csv(
        item_out_path, sep="\t", index=False, header=True, encoding="utf-8"
    )
    print(f"items saved in: {item_out_path.resolve()}")
    return str(item_out_path)
