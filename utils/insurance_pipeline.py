import pandas as pd
import numpy as np

def build_X_y(csv_path: str, include_bmi_smoker: bool = False):
    """
    Loads insurance CSV, applies standard cleaning + feature engineering,
    and returns (X, y, df_clean).

    Target: log_charges
    Features: age, bmi, children, smoker_flag + one-hot encoded region (drop_first)
    """
    df = pd.read_csv(csv_path)

    # --- cleaning / standardization ---
    df = df.drop_duplicates()
    

    for col in ['sex', 'smoker', 'region']:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df['smoker_flag'] = df['smoker'].map({'yes': 1, 'no': 0})
    if df['smoker_flag'].isna().any():
        bad_vals = df.loc[df['smoker_flag'].isna(), 'smoker'].unique()
        raise ValueError(f"Unexpected smoker values found: {bad_vals}")

    df['log_charges'] = np.log(df['charges'])

    # --- features / target ---
    y = df['log_charges']
    X = df[['age', 'bmi', 'children', 'smoker_flag']]

    # one-hot encode region (baseline dropped)
    X = pd.get_dummies(X.join(df['region']), drop_first=True)
    
    if include_bmi_smoker:
        X['bmi_smoker'] = df['bmi'] * df['smoker_flag']

    return X, y, df
