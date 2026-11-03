"""
E-Commerce Customer Segmentation & Recommendation Analysis
----------------------------------------------------------
Dataset: UCI Online Retail (Excel file)
Run:
    pip install -r requirements.txt
    python main.py --data Online_Retail.xlsx

If --data is omitted, the script attempts to download the public UCI dataset.
All generated files are saved in the outputs/ directory.
"""

import argparse
import os
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def load_data(path):
    """Load an Excel/CSV file, or download and extract the UCI Online Retail data."""
    if path and os.path.exists(path):
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path, encoding="latin1")

    os.makedirs("data", exist_ok=True)
    zip_path = os.path.join("data", "online_retail.zip")
    print("Downloading the UCI Online Retail dataset...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls"))]
        if not candidates:
            raise FileNotFoundError("No Excel dataset found inside the downloaded archive.")
        extracted = zf.extract(candidates[0], "data")
    return pd.read_excel(extracted)


def clean_transactions(df):
    """Clean transaction records and calculate line-item revenue."""
    required = {"InvoiceNo", "StockCode", "Description", "Quantity",
                "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = df.copy()
    data["CustomerID"] = data["CustomerID"].astype("string").str.strip()
    data = data.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate"])
    data["InvoiceNo"] = data["InvoiceNo"].astype(str)
    data = data[~data["InvoiceNo"].str.startswith("C", na=False)]
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate"])
    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    data["CustomerID"] = data["CustomerID"].astype(str)
    return data


def build_rfm(data):
    """Create Recency, Frequency and Monetary values per customer."""
    snapshot_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = data.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        TotalQuantity=("Quantity", "sum"),
    )
    return rfm, snapshot_date


def choose_k(scaled, min_k=2, max_k=8):
    """Calculate inertia and silhouette scores to help select cluster count."""
    rows = []
    upper = min(max_k, len(scaled) - 1)
    for k in range(min_k, upper + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels) if len(set(labels)) > 1 else np.nan
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": score})
    return pd.DataFrame(rows)


def assign_segment_profiles(rfm):
    """Give clusters understandable names using relative RFM averages."""
    summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    # High frequency and monetary value, together with low recency, indicate valuable customers.
    recency_rank = summary["Recency"].rank(ascending=True, pct=True)
    frequency_rank = summary["Frequency"].rank(ascending=True, pct=True)
    monetary_rank = summary["Monetary"].rank(ascending=True, pct=True)
    value_score = (1 - recency_rank) + frequency_rank + monetary_rank

    order = value_score.sort_values(ascending=False).index.tolist()
    names = {}
    labels = ["Champions", "Regular Customers", "Occasional Buyers", "At-Risk Customers"]
    for i, cluster_id in enumerate(order):
        names[cluster_id] = labels[i] if i < len(labels) else f"Customer Segment {i+1}"
    return names


def build_recommendations(data, customer_ids, top_n=5):
    """Recommend products by cosine similarity between customer purchase profiles."""
    basket = pd.crosstab(data["CustomerID"], data["Description"])
    # Binary purchase profile prevents high quantities from dominating similarity.
    basket = (basket > 0).astype(int)
    similarity = cosine_similarity(basket)
    sim_df = pd.DataFrame(similarity, index=basket.index, columns=basket.index)

    recommendations = {}
    for customer_id in customer_ids:
        if customer_id not in basket.index:
            continue
        similar_customers = sim_df[customer_id].drop(customer_id).sort_values(ascending=False).head(5)
        already_bought = set(basket.columns[basket.loc[customer_id] > 0])
        candidate_scores = pd.Series(dtype=float)
        for other_id, sim_score in similar_customers.items():
            products = basket.loc[other_id]
            unseen = products[(products > 0) & (~products.index.isin(already_bought))]
            if not unseen.empty:
                candidate_scores = candidate_scores.add(unseen.astype(float) * sim_score, fill_value=0)
        recommendations[customer_id] = candidate_scores.sort_values(ascending=False).head(top_n).index.tolist()
    return recommendations


def save_plots(rfm, evaluation, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(evaluation["k"], evaluation["inertia"], marker="o")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "elbow_method.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(evaluation["k"], evaluation["silhouette"], marker="o")
    plt.title("Silhouette Score by Cluster Count")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "silhouette_scores.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Segment", alpha=0.7)
    plt.title("Customer Segments: Frequency vs Monetary")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "customer_clusters.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.heatmap(rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean(),
                annot=True, fmt=".1f", cmap="Blues")
    plt.title("Average RFM Values by Segment")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segment_profile_heatmap.png"), dpi=150)
    plt.close()


def run(data_path=None, output_dir="outputs", k=None):
    os.makedirs(output_dir, exist_ok=True)
    raw = load_data(data_path)
    transactions = clean_transactions(raw)
    rfm, snapshot_date = build_rfm(transactions)

    # Log transform reduces the influence of highly skewed monetary/frequency values.
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features[["Frequency", "Monetary"]] = np.log1p(features[["Frequency", "Monetary"]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    evaluation = choose_k(scaled)
    evaluation.to_csv(os.path.join(output_dir, "cluster_evaluation.csv"), index=False)
    if k is None:
        best_k = int(evaluation.loc[evaluation["silhouette"].idxmax(), "k"])
        # Silhouette is a guide, not a substitute for business interpretation.
    else:
        best_k = int(k)
    if best_k < 2 or best_k >= len(rfm):
        raise ValueError("Choose a cluster count from 2 up to number of customers - 1.")

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(scaled)
    segment_names = assign_segment_profiles(rfm)
    rfm["Segment"] = rfm["Cluster"].map(segment_names)

    rfm.reset_index().to_csv(os.path.join(output_dir, "customer_segments.csv"), index=False)
    segment_summary = rfm.groupby("Segment").agg(
        Customers=("Cluster", "size"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
    ).round(2).sort_values("Customers", ascending=False)
    segment_summary.to_csv(os.path.join(output_dir, "segment_profiles.csv"))

    # Product-level purchase matrix and sample recommendations.
    recommendations = build_recommendations(transactions, rfm.index.tolist()[:100])
    with open(os.path.join(output_dir, "sample_recommendations.txt"), "w", encoding="utf-8") as f:
        for customer_id, products in recommendations.items():
            f.write(f"Customer {customer_id}:\\n")
            f.write("\\n".join(f"  - {p}" for p in products) if products else "  No recommendation available")
            f.write("\\n\\n")

    save_plots(rfm, evaluation, output_dir)
    print("\\nAnalysis completed.")
    print(f"Snapshot date: {snapshot_date.date()}")
    print(f"Customers analysed: {len(rfm):,}")
    print(f"Selected clusters: {best_k}")
    print("\\nCustomer segment summary:")
    print(segment_summary)
    print(f"\\nFiles saved in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer segmentation and product recommendation")
    parser.add_argument("--data", default=None, help="Path to Online_Retail.xlsx or CSV")
    parser.add_argument("--output", default="outputs", help="Output folder")
    parser.add_argument("--k", type=int, default=None, help="Optional number of clusters")
    args = parser.parse_args()
    run(args.data, args.output, args.k)
