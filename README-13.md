# E-Commerce Customer Segmentation & Recommendation Analysis

## Project overview

This project analyses customer purchasing behaviour from the **UCI
Online Retail** transaction dataset. It creates customer-level Recency,
Frequency and Monetary (RFM) features, applies K-Means clustering to
identify customer groups, and generates simple product recommendations
using cosine similarity.

## Objectives

-   Clean and prepare retail transaction data.
-   Build an RFM-based customer feature table.
-   Scale customer features and perform K-Means clustering.
-   Evaluate possible cluster counts using the Elbow method and
    silhouette score.
-   Describe clusters using business-friendly labels.
-   Recommend products based on the purchase patterns of similar
    customers.

## Technology stack

-   Python
-   Pandas and NumPy
-   Scikit-learn (StandardScaler, KMeans, cosine similarity)
-   Matplotlib and Seaborn
-   Jupyter Notebook (optional; the supplied implementation runs as a
    Python script)

## Dataset

Use the UCI Online Retail dataset. It contains invoice-level retail
transactions, including invoice number, product code and description,
quantity, invoice date, unit price, customer ID and country.

The script downloads the public UCI archive automatically when no local
dataset path is provided. Alternatively, download the dataset and place
`Online_Retail.xlsx` in the project folder.

## Project structure

``` text
ecommerce_customer_segmentation/
├── main.py
├── requirements.txt
├── README.md
├── REPORT.pdf
└── outputs/                 # Created after running the program
    ├── customer_segments.csv
    ├── segment_profiles.csv
    ├── cluster_evaluation.csv
    ├── sample_recommendations.txt
    ├── elbow_method.png
    ├── silhouette_scores.png
    ├── customer_clusters.png
    └── segment_profile_heatmap.png
```

## Setup and execution

### 1. Create and activate a virtual environment (recommended)

``` bash
python -m venv venv
```

Windows:

``` bash
venv\Scripts\activate
```

macOS/Linux:

``` bash
source venv/bin/activate
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Run the analysis

Automatic dataset download:

``` bash
python main.py
```

Using a local Excel file:

``` bash
python main.py --data Online_Retail.xlsx
```

Choose a cluster count manually:

``` bash
python main.py --data Online_Retail.xlsx --k 4
```

## Methodology

### 1. Data cleaning

Rows without a customer ID, invoice number or valid invoice date are
removed. Cancelled invoices (invoice numbers beginning with `C`),
non-positive quantities and non-positive unit prices are excluded.
Revenue is calculated as `Quantity × UnitPrice`.

### 2. RFM feature engineering

-   **Recency:** number of days since the customer's most recent
    purchase.
-   **Frequency:** number of distinct invoices/orders placed by the
    customer.
-   **Monetary:** total transaction revenue for the customer.

A snapshot date one day after the latest transaction is used to
calculate recency.

### 3. Preprocessing and clustering

Frequency and Monetary values are log-transformed to reduce skew. The
three RFM features are standardized and passed to K-Means. Candidate
cluster counts are evaluated using inertia and silhouette score. The
selected cluster count should also be checked for business usefulness.

### 4. Segment interpretation

Clusters are ranked using their average RFM characteristics and assigned
readable labels such as Champions, Regular Customers, Occasional Buyers
and At-Risk Customers. These labels are indicative and should be
reviewed against the exported segment profiles.

### 5. Recommendation component

A binary customer-product purchase matrix is created. Cosine similarity
finds customers with similar purchase histories. Products bought by
similar customers, but not yet purchased by the target customer, are
ranked as recommendations.

## Output files

-   `customer_segments.csv`: customer RFM values, cluster number and
    segment label.
-   `segment_profiles.csv`: average RFM values and customer count for
    each segment.
-   `cluster_evaluation.csv`: inertia and silhouette scores for
    candidate values of k.
-   `sample_recommendations.txt`: example product suggestions for up to
    100 customers.
-   PNG files: cluster evaluation and segment visualisations.

## Important notes and limitations

-   The automatic download requires an internet connection.
-   The recommendation method is a basic item-profile approach; it does
    not use product descriptions, ratings or deep learning.
-   Customers with sparse purchase histories may receive few or no
    recommendations.
-   The dataset is historical. Results describe the supplied dataset and
    should not be treated as a live customer profile.
-   Segment labels are heuristic. Inspect the CSV summaries before
    making business decisions.
-   The full customer-product similarity matrix can require substantial
    memory for very large datasets. This implementation is intended as
    an educational project on the UCI dataset.

## Future enhancements

-   Add a Streamlit dashboard with filters and interactive charts.
-   Compare K-Means with hierarchical clustering and DBSCAN.
-   Add item-item collaborative filtering and recommendation evaluation.
-   Schedule periodic data refreshes and monitor segment movement.
-   Include customer lifetime value and product category features.

## Author

Student Project -- E-Commerce Customer Segmentation & Recommendation
Analysis
