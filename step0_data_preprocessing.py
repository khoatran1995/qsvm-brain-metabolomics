import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def prepare_and_save(n_samples, filename):
    print(f"--- Preparing Data for {n_samples} Samples ---")
    
    # 1. Load raw data
    print("Loading raw feature_matrix.csv...")
    df = pd.read_csv("data/feature_matrix.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Subsample data
    df = df.sample(n=n_samples, random_state=42)
    
    # Encode target (Sex)
    le = LabelEncoder()
    y = le.fit_transform(df['Gender'])
    X = df.drop(columns=['Gender']).values
    X = np.nan_to_num(X) # Handle any missing values
    
    # 2. PCA Compression (Dimensionality Reduction)
    # We reduce the features to exactly 5 because we want to map this to 5 Qubits
    print("Compressing data to 5 Principal Components (Qubit Mapping)...")
    pca = PCA(n_components=5, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # 3. Angle Scaling (The Pi Trick)
    # Scale all features to be between 0 and Pi for quantum rotation encoding
    print("Scaling values between 0 and pi...")
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_pca)
    
    # 4. Split and Save
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    np.savez(f"data/{filename}", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(f"Saved quantum-ready data to data/{filename}: Train {X_train.shape}, Test {X_test.shape}\n")

# Run the pipeline for both the 100-sample and 150-sample experiments
prepare_and_save(100, "patient_data_100.npz")
prepare_and_save(150, "patient_data_150.npz")
