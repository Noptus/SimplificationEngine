import pandas as pd
import numpy as np
from hdbcli import dbapi
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# -----------------------------------------------------------------------------
# Database connection parameters - update these with your actual credentials.
HANA_HOST = 'hana_host_address'
HANA_PORT = 30015
HANA_USER = 'your_username'
HANA_PASSWORD = 'your_password'
TABLE_NAME = 'MATERIALS_TABLE'  # Update with your actual table name

# -----------------------------------------------------------------------------
def connect_to_hana():
    """
    Connects to the SAP HANA database using the hdbcli library.
    Returns the connection object if successful.
    """
    try:
        conn = dbapi.connect(
            address=HANA_HOST,
            port=HANA_PORT,
            user=HANA_USER,
            password=HANA_PASSWORD
        )
        print("Connection to SAP HANA established successfully.")
        return conn
    except Exception as e:
        print(f"Error connecting to SAP HANA: {e}")
        raise

# -----------------------------------------------------------------------------
def fetch_data(conn):
    """
    Fetches data from the specified table and loads it into a Pandas DataFrame.
    """
    try:
        query = f"SELECT * FROM {TABLE_NAME}"
        cursor = conn.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        cursor.close()
        print(f"Fetched {len(df)} rows from table {TABLE_NAME}.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise

# -----------------------------------------------------------------------------
def preprocess_data(df):
    """
    Preprocesses the DataFrame:
      - Standardizes numeric columns.
      - Converts a comma-separated 'tags' column into dummy variables (if exists).
    Returns the processed DataFrame and the list of numeric columns used for comparison.
    """
    df_processed = df.copy()
    
    # Identify numeric columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Standardize numeric columns using StandardScaler
    if numeric_cols:
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    # Process 'tags' column if present: split comma-separated tags and create dummy variables.
    if 'tags' in df_processed.columns:
        # Create a new column with list of tags (strip any extra spaces)
        df_processed['tags_list'] = df_processed['tags'].astype(str).apply(lambda x: [tag.strip() for tag in x.split(',')])
        # Get all unique tags across rows
        unique_tags = set(tag for tags in df_processed['tags_list'] for tag in tags)
        # Create binary/dummy columns for each unique tag
        for tag in unique_tags:
            df_processed[f"tag_{tag}"] = df_processed['tags_list'].apply(lambda tags: 1 if tag in tags else 0)
        # Include the new dummy columns in numeric features
        tag_cols = [f"tag_{tag}" for tag in unique_tags]
        numeric_cols.extend(tag_cols)
    
    return df_processed, numeric_cols

# -----------------------------------------------------------------------------
def calculate_similarity(features, threshold=0.95):
    """
    Calculates the cosine similarity matrix for the given feature matrix.
    Flags duplicate rows if their similarity is above the provided threshold.
    
    Returns a dictionary mapping a row index to a list of duplicate row indices.
    """
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(features)
    duplicates = {}
    
    # Iterate only over upper triangular matrix (excluding self-similarity)
    n_rows = similarity_matrix.shape[0]
    for i in range(n_rows):
        for j in range(i+1, n_rows):
            if similarity_matrix[i, j] >= threshold:
                duplicates.setdefault(i, []).append(j)
    
    return duplicates

# -----------------------------------------------------------------------------
def detect_anomalies(df, numeric_cols, z_threshold=3):
    """
    Detects anomalies in each numeric column using the z-score method.
    Returns a dictionary mapping each column to a list of row indices that are considered outliers.
    """
    anomalies = {}
    for col in numeric_cols:
        try:
            # Convert column to float and compute z-scores
            col_data = df[col].astype(float)
            z_scores = np.abs(stats.zscore(col_data, nan_policy='omit'))
            outlier_indices = np.where(z_scores > z_threshold)[0]
            if len(outlier_indices) > 0:
                anomalies[col] = outlier_indices.tolist()
        except Exception as e:
            print(f"Error processing column {col} for anomaly detection: {e}")
    return anomalies

# -----------------------------------------------------------------------------
def main():
    # Connect to HANA database
    try:
        conn = connect_to_hana()
    except Exception:
        return
    
    # Fetch data from the database
    try:
        df = fetch_data(conn)
    except Exception:
        conn.close()
        return
    
    # Preprocess the data
    df_processed, numeric_cols = preprocess_data(df)
    
    # Ensure there are features to work with
    if not numeric_cols:
        print("No numeric features available for similarity computation or anomaly detection.")
        conn.close()
        return
    
    # Create feature matrix for similarity computation
    features = df_processed[numeric_cols].values
    
    # Calculate similarity scores and flag potential duplicates
    similarity_threshold = 0.95  # Adjust this threshold based on domain knowledge
    duplicates = calculate_similarity(features, threshold=similarity_threshold)
    
    print("Potential duplicate entries (by row index):")
    for key, dup_list in duplicates.items():
        print(f"Row {key} duplicates: {dup_list}")
    
    # Detect anomalies (outliers) in each numeric column
    anomalies = detect_anomalies(df, numeric_cols)
    print("\nDetected anomalies (outlier indices per column):")
    for col, indices in anomalies.items():
        print(f"{col}: {indices}")
    
    # Close the database connection
    conn.close()

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
