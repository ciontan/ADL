import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def apply_smote(input_file="data/processed_data.csv", output_file="data/processed_data_smote.csv"):
    df = pd.read_csv(input_file)
    
    X = df.drop(columns=['DR'])
    y = df['DR']

    print("Class distribution before SMOTE:")
    print(y.value_counts())
    
    #? Convert string numbers with commas to float, not needed but precautionary
    X = X.applymap(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) and x.replace(',', '').replace('.', '').isdigit() else x)

    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    print(f"Class distribution after SMOTE: {y_res.value_counts()}")

    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)

    df_resampled = pd.DataFrame(X_res, columns=X.columns)
    df_resampled['DR'] = y_res  
    df_resampled.to_csv(output_file, index=False)

    print(f"Balanced data saved to {output_file}")

if __name__ == "__main__":
    apply_smote()
