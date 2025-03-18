import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(original_file="data/processed_data.csv", smote_file="data/processed_data_smote.csv"):
    # Load datasets
    original_df = pd.read_csv(original_file)
    smote_df = pd.read_csv(smote_file)

    # Get numerical features (excluding target column 'DR')
    feature_columns = [col for col in original_df.columns if col != 'DR']

    num_features = len(feature_columns)
    fig, axes = plt.subplots(num_features, 2, figsize=(10, 3 * num_features))

    for i, feature in enumerate(feature_columns):
        # Before SMOTE
        sns.histplot(original_df[feature], kde=True, ax=axes[i, 0], color='blue')
        axes[i, 0].set_title(f"Before SMOTE: {feature}")

        # After SMOTE
        sns.histplot(smote_df[feature], kde=True, ax=axes[i, 1], color='red')
        axes[i, 1].set_title(f"After SMOTE: {feature}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_feature_distributions()
