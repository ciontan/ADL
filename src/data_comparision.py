import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(original_file="data/processed_data.csv", smote_file="data/processed_data_smote.csv"):
    original_df = pd.read_csv(original_file)
    smote_df = pd.read_csv(smote_file)

    feature_columns = [col for col in original_df.columns if col != 'DR']

    num_features = len(feature_columns)
    plot_folder = "plots"
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    fig, axes = plt.subplots(num_features, 2, figsize=(10, 3 * num_features))

    for i, feature in enumerate(feature_columns):
     
        sns.histplot(original_df[feature], kde=True, ax=axes[i, 0], color='blue')
        axes[i, 0].set_title(f"Before SMOTE: {feature}")

     
        sns.histplot(smote_df[feature], kde=True, ax=axes[i, 1], color='red')
        axes[i, 1].set_title(f"After SMOTE: {feature}")

    plot_filename = os.path.join(plot_folder, "feature_distributions.png")
    plt.tight_layout()
    plt.savefig(plot_filename)  
    plt.close() 
    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    plot_feature_distributions()
