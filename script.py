import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import gower
import os
import openai
from scipy.spatial.distance import pdist, squareform
from fpdf import FPDF
import tempfile

st.set_page_config(page_title="Hana Cleanup Tool", layout="wide")

# --- App Introduction ---
st.markdown("""
### 👋 Welcome to the Hana Cleanup Tool, by Raphael Caillon

This tool is designed for **business users** to help simplify and clean up large master data tables, like material master records. You can:

1. **Upload a CSV file** or **connect directly to SAP HANA** to load your data.
2. **Cluster similar records** using AI-powered methods.
3. **Spot anomalies** that stand out from the rest.
4. **Interactively reassign or rename clusters**.
5. **Export a report** that can be shared with the master data team to apply the changes in your system.

> This tool is built to be intuitive for non-technical users, while powerful enough for data professionals.
""")

# --- Config ---
st.title("🧹 Hana Cleanup Tool")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("🔐 Enter your OpenAI API Key", type="password")

# --- Data Source Selection ---
data_source = st.sidebar.radio("Choose data source:", ["CSV Upload", "SAP HANA"])

if data_source == "CSV Upload":
    uploaded_file = st.file_uploader("Upload a CSV file to analyze and cluster", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # --- Select features to use ---
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect("Select numeric features for clustering:", numeric_cols, default=numeric_cols)

        # everything else involving 'df' stays indented under this check
    

    # --- Set Weights ---
    if 'selected_features' in locals() and selected_features:
        st.sidebar.subheader("Feature Weights")
        feature_weights = {}
        selected_weight_feature = st.sidebar.selectbox("Select a feature to assign weight:", selected_features)
        default_weights = {col: 1.0 for col in selected_features}
        if 'feature_weights' not in st.session_state:
            st.session_state.feature_weights = default_weights
        weight = st.sidebar.slider("Weight", min_value=0.0, max_value=5.0, value=st.session_state.feature_weights[selected_weight_feature], step=0.1)
        st.session_state.feature_weights[selected_weight_feature] = weight
        feature_weights = st.session_state.feature_weights
    else:
        feature_weights = {}

    selected_features = selected_features if 'selected_features' in locals() else []
    mode = st.sidebar.radio("Select Mode:", ["Clustering", "Anomaly Spotter"])

    if mode == "Anomaly Spotter":
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if not cat_cols:
            st.warning("No categorical columns found for anomaly detection.")
        else:
            cat_col = st.selectbox("Select categorical column to group by:", cat_cols)
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected_metric = st.selectbox("Numeric column to detect anomalies in:", num_cols)

            outliers = []
            summaries = []
            st.subheader("🔎 Anomaly Detection by Category")
            for group, group_df in df.groupby(cat_col):
                mean = group_df[selected_metric].mean()
                std = group_df[selected_metric].std()
                summaries.append(f"Group '{group}': mean={mean:.2f}, std={std:.2f}")
                for idx, val in group_df[selected_metric].items():
                    z = (val - mean) / std if std > 0 else 0
                    if abs(z) > 3:
                        outliers.append((idx, group, val, round(z, 2)))

            st.markdown("### 📋 Group Summaries")
            for summary in summaries:
                st.markdown(f"- {summary}")

            if outliers:
                outlier_df = pd.DataFrame(outliers, columns=['Index', cat_col, selected_metric, 'Z-score'])
                st.write(f"Found {len(outliers)} outliers:")
                st.dataframe(outlier_df)

                st.subheader("📊 Outlier Deviation Plot")
                fig_out = px.scatter(outlier_df, x='Index', y='Z-score', color=cat_col,
                                     title="Z-score Deviation of Outliers")
                st.plotly_chart(fig_out, use_container_width=True)
            else:
                st.write("No outliers detected.")

    elif mode == "Clustering" and 'df' in locals() and selected_features and len(selected_features) >= 2:
        X = df[selected_features].dropna()
        for col in selected_features:
            X[col] = X[col] * feature_weights[col]

        st.sidebar.header("Clustering Settings")

        distance_metric = st.sidebar.radio("Select Distance Metric:", options=["Euclidean", "Gower"])

        if distance_metric == "Gower":
            distance_matrix = gower.gower_matrix(X)
            best_score = -1
            best_k = 2
            for k in range(2, min(11, len(X))):
                model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
                labels = model.fit_predict(distance_matrix)
                score = silhouette_score(distance_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_k = k
            model = AgglomerativeClustering(n_clusters=best_k, metric='precomputed', linkage='average')
            cluster_labels = model.fit_predict(distance_matrix)
            X_scaled = distance_matrix
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            best_score = -1
            best_k = 2
            for k in range(2, min(11, len(X))):
                model = KMeans(n_clusters=k, random_state=42)
                labels = model.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)

        df['cluster'] = cluster_labels

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled if distance_metric == "Euclidean" else X)
        df['x'] = X_pca[:, 0]
        df['y'] = X_pca[:, 1]

        st.subheader("🖱️ Cluster Interaction")

        fig = px.scatter(df, x='x', y='y', color=df['cluster'].astype(str),
                         hover_data=selected_features + ['cluster'],
                         title=f"📊 PCA Cluster Projection ({distance_metric} Distance, k={best_k})")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Reassign Cluster for a Data Point")
        selected_index = st.number_input("Enter index of point to reassign:", min_value=0, max_value=len(df)-1, step=1)
        current_cluster = df.at[selected_index, 'cluster']
        st.write(f"Current cluster: {current_cluster}")
        new_cluster = st.selectbox("Assign to new cluster:", options=[int(i) for i in range(best_k)], index=int(current_cluster))
        if st.button("Apply Reassignment"):
            df.at[selected_index, 'cluster'] = new_cluster
            st.success(f"Point {selected_index} moved from cluster {current_cluster} to {new_cluster}")

        st.subheader("📈 Cluster Summary")
        cluster_summary = df.groupby('cluster')[selected_features].agg(['mean', 'std', 'min', 'max'])
        cluster_summary.columns = ['_'.join(col) for col in cluster_summary.columns]
        st.dataframe(cluster_summary.style.format("{:.2f}"))

        st.subheader("📦 Cluster Sizes")
        cluster_sizes = df['cluster'].value_counts().sort_index()
        st.bar_chart(cluster_sizes)

        try:
            st.metric(label="Silhouette Score", value=f"{best_score:.3f}", delta=None)
        except Exception as e:
            st.warning("Couldn't compute silhouette score: " + str(e))

        st.subheader("🔍 Exact Duplicates")
        duplicates = df[df.duplicated(selected_features, keep=False)]
        if not duplicates.empty:
            st.write(f"Found {len(duplicates)} exact duplicate entries:")
            st.dataframe(duplicates)
        else:
            st.write("No exact duplicates found.")

        ai_response = ""
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            try:
                with st.spinner("🤖 Naming clusters with AI..."):
                    features_summary = cluster_summary.reset_index().to_dict()
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a data simplification assistant."},
                            {"role": "user", "content": f"Given this cluster summary: {features_summary}, please name each cluster with a descriptive title and briefly summarize what each one represents."}
                        ]
                    )
                    ai_response = response.choices[0].message.content if hasattr(response.choices[0], 'message') else response.choices[0].text
                    st.subheader("🧠 AI Interpretation of Clusters")
                    st.markdown(ai_response)
            except Exception as e:
                st.error(f"AI request failed: {str(e)}")

        # Export PDF summary
        if st.button("📄 Export PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Hana Cleanup Tool Report", ln=True, align='C')
            pdf.cell(200, 10, txt=f"Silhouette Score: {best_score:.3f}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Number of Clusters: {best_k}", ln=True, align='L')
            pdf.ln(10)
            pdf.set_font("Arial", size=10)
            for cluster_id, row in cluster_summary.iterrows():
                pdf.cell(200, 10, txt=f"Cluster {cluster_id}", ln=True)
                for feature, value in row.items():
                    pdf.cell(200, 8, txt=f" - {feature}: {round(value, 2)}", ln=True)
                pdf.ln(5)
            if ai_response:
                pdf.ln(10)
                pdf.multi_cell(0, 8, txt="AI Interpretation:\n" + ai_response)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                pdf.output(tmp_file.name)
                with open(tmp_file.name, "rb") as f:
                    st.download_button(label="Download PDF Summary", data=f, file_name="hana_cleanup_report.pdf", mime="application/pdf")

        csv_out = df.copy()
        csv_out.to_csv("clustered_output.csv", index=False)
        st.download_button("Download Clustered Data", data=csv_out.to_csv(index=False),
                           file_name="clustered_output.csv", mime="text/csv")
elif data_source == "SAP HANA":
    st.subheader("🔌 Connect to SAP HANA")
    with st.form("hana_form"):
        hana_host = st.text_input("HANA Host")
        hana_user = st.text_input("HANA User")
        hana_password = st.text_input("HANA Password", type="password")
        table_name = st.text_input("Table Name")
        connect = st.form_submit_button("Connect")

    if connect:
        try:
            from hdbcli import dbapi
            conn = dbapi.connect(address=hana_host, port=30015, user=hana_user, password=hana_password)
            query = f"SELECT * FROM \"{table_name}\""
            df = pd.read_sql(query, conn)
            st.success(f"Connected and loaded table: {table_name}")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Connection failed: {e}")
else:
    st.info("Please upload a CSV file or connect to a SAP HANA table to begin.")
