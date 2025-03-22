import pandas as pd
from sklearn.preprocessing import StandardScaler

def gender_to_ones_and_zeros(df):
    if df["Gender"].isin([0, 1]).all():
        print("gender column already in 0 and 1")
        return df
    df[["Gender"]] = df[["Gender"]].replace({1: 0, 2: 1})
    return df

def load_and_preprocess_data(input_file="data/raw_data.csv", output_file="data/processed_data.csv", normalise=True):
    """input_file: str, path to the raw data file
    output_file: str, path to save the processed data file"""
    df = pd.read_csv(input_file)
    
    df.rename(columns={'age': 'Age','gender': 'Gender', 'community': 'Community', 'U-Alb': 'UAlb', 
                   'LDL-C': 'LDLC', 'HDL-C': 'HDLC', 'ACR': 'UACR'}, inplace=True)
    continuous_features = ['Age', 'UAlb', 'Ucr', 'UACR', 'TC', 'TG', 'TCTG', 'LDLC', 'HDLC', 
                           'Scr', 'BUN', 'FPG', 'HbA1c', 'Height', 'Weight', 'BMI', 'Duration']
    
    #? converting numbers like 1,203.45 to 1203.45
    for col in continuous_features:
        df[col] = df[col].astype(str).str.replace(',', '')  
        df[col] = pd.to_numeric(df[col], errors='coerce') 
    
    #? filling missing values with median, although there are no missing values for continuous features
    df[continuous_features] = df[continuous_features].fillna(df[continuous_features].median()) #not really need but ok
   
    #? slicing only what we need, the 19 features
    df = df[["Age", 
            "Gender", 
            "Community", 
            "UAlb", 
            "Ucr", 
            "UACR", 
            "TC", 
            "TG", 
            "TCTG", 
            "LDLC", 
            "HDLC", 
            "Scr", 
            "BUN", 
            "FPG", 
            "HbA1c", 
            "Height", 
            "Weight", 
            "BMI", 
            "Duration",
            "DR"]]
    #? one hot encoding community, 10 communities, do not "drop first = True"
    df = pd.get_dummies(df, columns=['Community'], dtype = float) 
    
    #? Converting 1s and 2s into 0s and 1s in place
    df = gender_to_ones_and_zeros(df)
    
        #? Normalising for continuous features
    if normalise:
        scaler = StandardScaler()
        df[continuous_features] = scaler.fit_transform(df[continuous_features])
    df.to_csv(output_file, index=False)

    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    load_and_preprocess_data()