import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(input_file="data/raw_data.csv", output_file="data/processed_data.csv"):
    df = pd.read_csv(input_file)
    df.rename(columns={'gender': 'Gender', 'community': 'Community', 'U-Alb': 'UAlb', 
                   'LDL-C': 'LDLC', 'HDL-C': 'HDLC', 'ACR': 'UACR'}, inplace=True)
    df = df.dropna() 

    continuous_features = ['age', 'UAlb', 'Ucr', 'UACR', 'TC', 'TG', 'LDLC', 'HDLC', 
                           'Scr', 'BUN', 'FPG', 'HbA1c', 'Height', 'Weight', 'BMI']
    
    for col in continuous_features:
        df[col] = df[col].astype(str).str.replace(',', '')  
        df[col] = pd.to_numeric(df[col], errors='coerce') 
    
    df[continuous_features] = df[continuous_features].fillna(df[continuous_features].median())
    df = pd.get_dummies(df, columns=['Gender', 'Community'], drop_first=True)
    
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])
    df.to_csv(output_file, index=False)

    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    load_and_preprocess_data()