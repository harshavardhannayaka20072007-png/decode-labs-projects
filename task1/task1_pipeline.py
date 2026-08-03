import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("==========================================")
print("--- LOADING DATASET ---")
print("==========================================")
df = pd.read_csv('dataset.csv')
df_module1 = df.copy()

# ==========================================
# 2. MODULE 1: INPUT (DATA FIDELITY)
# ==========================================
print("\n==========================================")
print("--- MODULE 1: INPUT & CLEANING ---")
print("==========================================")

# Step 1A: Handle < 5% missingness by dropping rows upfront
missing_pct = df_module1.isnull().mean()
low_missing_cols = missing_pct[(missing_pct > 0) & (missing_pct < 0.05)].index.tolist()

if low_missing_cols:
    df_module1 = df_module1.dropna(subset=low_missing_cols)
    print(f"Dropped rows for < 5% missing columns: {low_missing_cols}")

# Recalculate missing percentages after row drops
missing_pct = df_module1.isnull().mean()

# Step 1B: Handle 5% - 20% (Median / Mode Imputation)
for col in df_module1.columns:
    pct = missing_pct[col]
    if 0.05 <= pct <= 0.20:
        if pd.api.types.is_numeric_dtype(df_module1[col]):
            median_val = df_module1[col].median()
            df_module1[col] = df_module1[col].fillna(median_val)
            print(f"Column '{col}' (5-20% missing): Imputed with Global Median ({median_val}).")
        else:
            mode_val = df_module1[col].mode()[0]
            df_module1[col] = df_module1[col].fillna(mode_val)
            print(f"Column '{col}' (5-20% missing): Imputed with Mode ({mode_val}).")

# Step 1C: Handle > 20% Missingness (Multi-Dimensional KNN Imputation)
high_missing_num_cols = [
    col for col in df_module1.columns 
    if missing_pct[col] > 0.20 and pd.api.types.is_numeric_dtype(df_module1[col])
]

if high_missing_num_cols:
    all_num_cols = df_module1.select_dtypes(include=[np.number]).columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    df_module1[all_num_cols] = imputer.fit_transform(df_module1[all_num_cols])
    print(f"Columns {high_missing_num_cols} (> 20% missing): Applied multi-dimensional KNN Imputation.")

# Handle remaining categorical columns > 20%
for col in df_module1.columns:
    if missing_pct[col] > 0.20 and not pd.api.types.is_numeric_dtype(df_module1[col]):
        mode_cat = df_module1[col].mode()[0]
        df_module1[col] = df_module1[col].fillna(mode_cat)
        print(f"Categorical Column '{col}' (> 20% missing): Imputed with Category Mode ({mode_cat}).")

print("\n--- OUTLIER WINSORIZATION (IQR) ---")

numeric_cols = df_module1.select_dtypes(include=[np.number]).columns.tolist()

for num_col in numeric_cols:
    Q1 = df_module1[num_col].quantile(0.25)
    Q3 = df_module1[num_col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_before = ((df_module1[num_col] < lower_bound) | (df_module1[num_col] > upper_bound)).sum()
    df_module1[num_col] = np.clip(df_module1[num_col], lower_bound, upper_bound)
    print(f"Feature '{num_col}': IQR Boundary [{lower_bound:.2f}, {upper_bound:.2f}] | Capped {outliers_before} outliers.")


# ==========================================
# 3. MODULE 2: PROCESS (VECTORIZED COMPUTATION ENGINE)
# ==========================================
print("\n==========================================")
print("--- MODULE 2: PROCESS ENGINE ---")
print("==========================================")

df_module2 = df_module1.copy()

# Step 2A: Feature Engineering (Creating 3 Predictive Vectorized Features)
print("[Step 2A] Creating 3 Derived Features (Vectorized Operations)...")

if 'TotalPrice' in df_module2.columns and 'Quantity' in df_module2.columns:
    df_module2['Effective_UnitPrice'] = np.where(df_module2['Quantity'] > 0, df_module2['TotalPrice'] / df_module2['Quantity'], 0)

if 'Quantity' in df_module2.columns:
    q_threshold = df_module2['Quantity'].quantile(0.75)
    df_module2['Is_Bulk_Order'] = (df_module2['Quantity'] > q_threshold).astype(int)

if 'TotalPrice' in df_module2.columns:
    df_module2['Log_TotalPrice'] = np.log1p(df_module2['TotalPrice'])

print("Added features: 'Effective_UnitPrice', 'Is_Bulk_Order', 'Log_TotalPrice'")

# Step 2B: Categorical Encoding (One-Hot Encoding for Nominal Variables)
print("\n[Step 2B] Performing One-Hot Encoding...")

# Drop unique identifier/date columns that don't add predictive value
id_cols_to_drop = ['OrderID', 'Date', 'CustomerID', 'TrackingNumber']
df_module2 = df_module2.drop(columns=[c for c in id_cols_to_drop if c in df_module2.columns])

# Select remaining low-cardinality categorical columns
categorical_cols = df_module2.select_dtypes(include=['object', 'category']).columns.tolist()

if categorical_cols:
    df_module2 = pd.get_dummies(df_module2, columns=categorical_cols, drop_first=True, dtype=int)
    print(f"Encoded categorical features: {categorical_cols}")
# Step 2C: Multicollinearity Eradication (|r| > 0.80)
print("\n[Step 2C] Multicollinearity Eradication (|r| > 0.80)...")
target_col = 'TotalPrice' if 'TotalPrice' in df_module2.columns else df_module2.columns[-1]

num_df = df_module2.select_dtypes(include=[np.number])
corr_matrix = num_df.corr().abs()

pairs_to_check = []
cols = [c for c in corr_matrix.columns if c != target_col]

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        col1, col2 = cols[i], cols[j]
        if corr_matrix.loc[col1, col2] > 0.80:
            pairs_to_check.append((col1, col2, corr_matrix.loc[col1, col2]))

cols_to_drop = set()

for col1, col2, corr_val in pairs_to_check:
    if col1 in cols_to_drop or col2 in cols_to_drop:
        continue
    
    corr_target_1 = abs(df_module2[col1].corr(df_module2[target_col]))
    corr_target_2 = abs(df_module2[col2].corr(df_module2[target_col]))
    
    if corr_target_1 >= corr_target_2:
        cols_to_drop.add(col2)
        print(f"High Correlation ({corr_val:.2f}) between '{col1}' and '{col2}'. Dropping weaker predictor: '{col2}'")
    else:
        cols_to_drop.add(col1)
        print(f"High Correlation ({corr_val:.2f}) between '{col1}' and '{col2}'. Dropping weaker predictor: '{col1}'")

df_module2 = df_module2.drop(columns=list(cols_to_drop))


# ==========================================
# 4. MODULE 3: OUTPUT (SCHEMA VALIDATION & SERVING)
# ==========================================
print("\n==========================================")
print("--- MODULE 3: CONTRACTS & SERVING FRAMEWORK ---")
print("==========================================")

df_module3 = df_module2.copy()

# Step 3A: Add Timestamp Identifier (Feature Store Concept - Point-in-time compliance)
df_module3['event_timestamp'] = pd.Timestamp.now()

# Step 3B: Pandera Schema Validation with lazy=True
print("[Step 3B] Validating Schema with Pandera...")

# Build dynamic schema columns based on resulting processed DataFrame
schema_dict = {
    "event_timestamp": Column(pa.DateTime, nullable=False),
}

for col in df_module3.columns:
    if col == "event_timestamp":
        continue
    if pd.api.types.is_numeric_dtype(df_module3[col]):
        schema_dict[col] = Column(pa.Float, nullable=False, coerce=True)
    else:
        schema_dict[col] = Column(pa.String, nullable=False)

# Create Schema Object (WITHOUT lazy=True here)
schema = pa.DataFrameSchema(schema_dict)

try:
    # Pass lazy=True inside validate()
    validated_df = schema.validate(df_module3, lazy=True)
    print("Schema validation PASSED successfully! All constraints satisfied.")
except pa.errors.SchemaErrors as err:
    print("Schema validation failed with lazy evaluation. Logging errors:")
    print(err.failure_cases)
    validated_df = df_module3

# Step 3C: Export Production-Ready Clean Dataset
output_parquet = 'cleaned_dataset.parquet'
output_csv = 'cleaned_dataset.csv'

validated_df.to_parquet(output_parquet, index=False)
validated_df.to_csv(output_csv, index=False)

print(f"\nPipeline Execution Finished Successfully!")
print(f"Cleaned dataset saved as:\n - Parquet: '{output_parquet}'\n - CSV: '{output_csv}'")

# ==========================================
# 5. MODULE 4: GRAPHICAL VISUALIZATION
# ==========================================
print(
    "\n=========================================="
)
print("--- GENERATING VISUALIZATIONS ---")
print(
    "=========================================="
)

# Set visual style
sns.set_theme(style="whitegrid")

# Create a figure with subplots
plt.figure(figsize=(10, 5))

# Plot 1: Distribution of TotalPrice after cleaning
plt.subplot(1, 2, 1)
sns.histplot(validated_df["TotalPrice"], kde=True, color="blue")
plt.title("Distribution of TotalPrice")

# Plot 2: Boxplot to check for remaining outliers
plt.subplot(1, 2, 2)
sns.boxplot(y=validated_df["TotalPrice"], color="orange")
plt.title("Boxplot of TotalPrice")

plt.tight_layout()

# Save the plot as an image so you can use it in your reports/LinkedIn
output_plot = "task1_visualization.png"
plt.savefig(output_plot)
print(f"Visualization saved successfully as: '{output_plot}'")

# Display the interactive plot window
plt.show()