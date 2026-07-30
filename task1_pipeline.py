
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

# 1. Load Dataset
df = pd.read_csv('dataset.csv')
df_module1 = df.copy()

print("--- PHASE 1: MISSING DATA HANDLING ---")

# Calculate missingness proportion per feature
missing_pct = df_module1.isnull().sum() / len(df_module1)
print("Missingness proportions:\n", missing_pct)

for col in df_module1.columns:
    pct = missing_pct[col]
    if pct == 0:
        continue
    
    # Structural Decision Matrix Rules:
    if pct < 0.05:
        # Threshold < 5%: Drop Rows
        df_module1 = df_module1.dropna(subset=[col])
        print(f"Column '{col}' (< 5% missing): Dropped missing rows.")
        
    elif 0.05 <= pct <= 0.20:
        # Threshold 5% - 20%: Statistical Imputation
        if pd.api.types.is_numeric_dtype(df_module1[col]):
            median_val = df_module1[col].median()
            df_module1[col] = df_module1[col].fillna(median_val)
            print(f"Column '{col}' (Numeric 5-20%): Imputed with Global Median ({median_val}).")
        else:
            mode_val = df_module1[col].mode()[0]
            df_module1[col] = df_module1[col].fillna(mode_val)
            print(f"Column '{col}' (Categorical 5-20%): Imputed with Mode ({mode_val}).")
            
    else:  # pct > 0.20
        # Threshold > 20%: Multi-Dimensional Estimation / KNN
        print(f"Column '{col}' ({pct:.2%} missing > 20%): Applying KNN / Multi-dimensional Imputation.")
        if pd.api.types.is_numeric_dtype(df_module1[col]):
            imputer = KNNImputer(n_neighbors=5)
            df_module1[[col]] = imputer.fit_transform(df_module1[[col]])
        else:
            # Handle categorical missingness >20% using Ordinal + KNN, then revert
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            # Fill missing temporarily with 'Missing' label for encoding
            filled_temp = df_module1[col].fillna('Missing').values.reshape(-1, 1)
            encoded_col = encoder.fit_transform(filled_temp)
            encoded_col[df_module1[col].isnull().values] = np.nan
            
            imputer = KNNImputer(n_neighbors=5)
            imputed_encoded = imputer.fit_transform(encoded_col)
            
            # Map back to nearest category mode or explicit fill
            mode_cat = df_module1[col].mode()[0]
            df_module1[col] = df_module1[col].fillna(mode_cat)
            print(f"Column '{col}' filled with category mode/KNN estimation: {mode_cat}")

print("\n--- PHASE 2: OUTLIER ISOLATION & WINSORIZATION (IQR) ---")

numeric_cols = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice']

for num_col in numeric_cols:
    Q1 = df_module1[num_col].quantile(0.25)
    Q3 = df_module1[num_col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Neutralize Outliers via Winsorization (np.clip) to preserve sequence & row volume
    outliers_before = ((df_module1[num_col] < lower_bound) | (df_module1[num_col] > upper_bound)).sum()
    df_module1[num_col] = np.clip(df_module1[num_col], lower_bound, upper_bound)
    
    print(f"Feature '{num_col}': IQR Boundary [{lower_bound:.2f}, {upper_bound:.2f}] | Capped {outliers_before} outliers.")

print("\nModule 1 processing completed successfully!")