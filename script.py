import streamlit as st
import pandas as pd
import numpy as np
import os
import openai
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from fpdf import FPDF
import gower

st.set_page_config(page_title="Hana Cleanup Tool", layout="wide")
st.title("🧹 Hana Cleanup Tool")

st.markdown("""
### 👋 Welcome to the Hana Cleanup Tool

1. Load your master data (CSV or SAP HANA)  
2. Detect duplicates  
3. Cluster similar records  
4. Spot anomalies  
5. Enrich data with AI  
6. Export a report to your Master Data Governance team  
""")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("🔐 Enter OpenAI API Key", type="password")
data_source = st.sidebar.radio("Choose data source:", ["CSV Upload", "SAP HANA"])
df = None

def load_csv(file):
    return pd.read_csv(file)

def connect_to_hana(host, user, password, table_name):
    df = pd.DataFrame({"material": ["A", "B", "C"], "value": [10, 20, 30]})
    return df, "Connected to HANA (dummy)"

def compute_feature_weighted_matrix(df, features, weights):
    X = df[features].copy()
    for col in features:
        X[col] *= weights[col]
    return X

def auto_cluster(X, method):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_k = 3
    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(X_scaled)
    labels = kmeans.labels_
    best_score = kmeans.inertia_
    return labels, best_k, best_score, X_scaled

def compute_pca_projection(X):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    return coords

def detect_anomalies(df, group_col, measure_col):
    summaries = []
    outliers = []
    for group, subset in df.groupby(group_col):
        mean = subset[measure_col].mean()
        std = subset[measure_col].std()
        summaries.append((group, mean, std))
        outliers.append(subset[np.abs(subset[measure_col] - mean) > 2 * std])
    outlier_df = pd.concat(outliers)
    return summaries, outlier_df

def generate_pdf(cluster_summary, best_score, best_k, comments, ai_response, include_dups, dups, include_clusters, include_outliers, outliers):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Master Data Report", ln=True, align='C')
    pdf.multi_cell(0, 10, txt=f"Comments: {comments}\n\nAI Summary: {ai_response}")
    if include_clusters and cluster_summary is not None:
        pdf.multi_cell(0, 10, txt="\nCluster Summary:\n" + cluster_summary.to_string())
    if include_dups and not dups.empty:
        pdf.multi_cell(0, 10, txt="\nDuplicates:\n" + dups.to_string())
    if include_outliers and not outliers.empty:
        pdf.multi_cell(0, 10, txt="\nOutliers:\n" + outliers.to_string())
    path = "report.pdf"
    pdf.output(path)
    return path

def offer_pdf_download(path):
    with open(path, "rb") as file:
        st.download_button(label="Download Report", data=file, file_name="report.pdf")

def compute_gower_distance(df, features):
    return gower.gower_matrix(df[features])

def find_gower_duplicates(df, features, threshold):
    distances = compute_gower_distance(df, features)
    duplicate_indices = set()
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            if distances[i][j] < threshold:
                duplicate_indices.add(i)
                duplicate_indices.add(j)
    return df.iloc[list(duplicate_indices)]

def enrich_data_with_ai(data):
    try:
        st.subheader("🤖 AI-Based Enrichment")
        if not OPENAI_API_KEY:
            st.warning("Please enter your OpenAI API key to enrich the data.")
            return data

        openai.api_key = OPENAI_API_KEY
        mode = st.radio("Choose enrichment mode:", ["Fill missing values", "Create synthetic column"])

        if mode == "Fill missing values":
            text_cols = data.select_dtypes(include='object').columns.tolist()
            if not text_cols:
                st.info("No text columns available.")
                return data
            col_to_fill = st.selectbox("Select column with missing values", text_cols)
            prompt = st.text_area("AI Prompt for filling missing data", "Fill missing entries with contextual completion.")
            missing_idx = data[col_to_fill].isna()
            if missing_idx.sum() > 0 and st.button("Run AI Fill"):
                with st.spinner("Filling missing data using AI..."):
                    for idx in data[missing_idx].index:
                        context = data.loc[idx].drop(col_to_fill).to_dict()
                        msg = f"{prompt}\nContext: {context}"
                        try:
                            response = openai.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant for data completion."},
                                    {"role": "user", "content": msg}
                                ]
                            )
                            filled_value = response.choices[0].message.content.strip()
                            data.at[idx, col_to_fill] = filled_value
                        except Exception as e:
                            st.error(f"Error at index {idx}: {e}")
                st.success("Missing values filled.")

        elif mode == "Create synthetic column":
            prompt = st.text_area("Describe the content of the synthetic column", "Generate a sales motivation score based on other fields.")
            col_name = st.text_input("Name for the new column", "synthetic_column")
            if st.button("Generate Column"):
                with st.spinner("Generating synthetic column..."):
                    new_col = []
                    for _, row in data.iterrows():
                        row_context = row.to_dict()
                        try:
                            response = openai.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "You generate synthetic business data."},
                                    {"role": "user", "content": f"{prompt}\nContext: {row_context}"}
                                ]
                            )
                            new_col.append(response.choices[0].message.content.strip())
                        except Exception as e:
                            new_col.append("Error")
                            st.error(f"Error on row: {e}")
                    data[col_name] = new_col
                    st.success(f"Synthetic column '{col_name}' added.")

            if col_name in data.columns:
                st.data_editor(data[[col_name]], num_rows="dynamic")

        return data
    except Exception as e:
        st.error(f"Error during enrichment: {e}")
        return data

if data_source == "CSV Upload":
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = load_csv(uploaded_file)
elif data_source == "SAP HANA":
    with st.form("hana_form"):
        hana_host = st.text_input("HANA Host")
        hana_user = st.text_input("HANA User")
        hana_password = st.text_input("HANA Password", type="password")
        table_name = st.text_input("Table Name")
        connect = st.form_submit_button("Connect")
        if connect:
            df, msg = connect_to_hana(hana_host, hana_user, hana_password, table_name)
            st.success(msg) if df is not None else st.error(msg)

if df is not None:
    step = st.sidebar.radio("Step", ["Find Duplicates", "Cluster Records", "Spot Outliers", "Enrich with AI", "Export Report"])
    selected_features = df.select_dtypes(include=[np.number]).columns.tolist()

    if step == "Find Duplicates":
        st.subheader("🔍 Duplicates")
        selected_features = st.multiselect("Select features to check duplicates", df.columns.tolist(), default=selected_features)
        threshold = st.slider("Similarity threshold (lower = more tolerant)", 0.0, 1.0, 0.2, 0.01)
        if st.button("Run Duplicate Detection"):
            with st.spinner("Calculating duplicates using Gower distance..."):
                try:
                    duplicates = find_gower_duplicates(df, selected_features, threshold)
                    if not duplicates.empty:
                        st.dataframe(duplicates)
                    else:
                        st.info("No duplicates found with this threshold.")
                except Exception as e:
                    st.error(f"Error computing duplicates: {e}")

    elif step == "Cluster Records":
        st.subheader("📊 Clustering")
        selected_features = st.multiselect("Select features", selected_features, default=selected_features)
        weights = {col: st.slider(f"Weight for {col}", 0.0, 5.0, 1.0, 0.1) for col in selected_features}
        X = compute_feature_weighted_matrix(df, selected_features, weights)
        method = st.radio("Distance Method", ["Euclidean", "Gower"])
        labels, best_k, best_score, X_scaled = auto_cluster(X, method)
        df['cluster'] = labels
        coords = compute_pca_projection(X_scaled)
        df['x'], df['y'] = coords[:, 0], coords[:, 1]
        st.plotly_chart(px.scatter(df, x='x', y='y', color=df['cluster'].astype(str)))
        cluster_summary = df.groupby('cluster')[selected_features].agg(['mean', 'std'])
        st.dataframe(cluster_summary)

    elif step == "Spot Outliers":
        st.subheader("🔎 Outlier Detection")
        cat_col = st.selectbox("Group by", df.select_dtypes(include='object').columns)
        num_col = st.selectbox("Measure", df.select_dtypes(include=np.number).columns)
        summaries, outlier_df = detect_anomalies(df, cat_col, num_col)
        [st.write(f"{g}: mean={m:.2f}, std={s:.2f}") for g, m, s in summaries]
        st.dataframe(outlier_df) if not outlier_df.empty else st.info("No outliers found.")

    elif step == "Enrich with AI":
        df = enrich_data_with_ai(df)

    elif step == "Export Report":
        st.subheader("📝 Export Report")
        comments = st.text_area("Comments for MDG team")
        flags = {
            "duplicates": st.checkbox("Include duplicates", True),
            "clusters": st.checkbox("Include clusters", True),
            "outliers": st.checkbox("Include outliers", True)
        }

        ai_response = ""
        if OPENAI_API_KEY and flags["clusters"]:
            try:
                openai.api_key = OPENAI_API_KEY
                with st.spinner("AI summarizing clusters..."):
                    cluster_summary = df.groupby('cluster')[selected_features].agg(['mean', 'std'])
                    summary_data = cluster_summary.reset_index().to_dict()
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a data simplification assistant."},
                            {"role": "user", "content": f"Summarize this cluster data: {summary_data}"}
                        ]
                    )
                    ai_response = response.choices[0].message.content
                    st.markdown(ai_response)
            except Exception as e:
                st.error(f"AI error: {e}")

        if st.button("📄 Generate PDF"):
            cluster_summary = df.groupby('cluster')[selected_features].agg(['mean', 'std']) if flags["clusters"] else None
            dups = df[df.duplicated(selected_features, keep=False)] if flags["duplicates"] else pd.DataFrame()
            outliers = outlier_df if flags["outliers"] and 'outlier_df' in locals() else pd.DataFrame()
            path = generate_pdf(cluster_summary, best_score if 'best_score' in locals() else 0.0,
                                best_k if 'best_k' in locals() else 0, comments, ai_response,
                                flags["duplicates"], dups, flags["clusters"], flags["outliers"], outliers)
            offer_pdf_download(path)
else:
    st.info("Upload a CSV or connect to HANA to get started.")
