import os
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
    plot_folder = "plots"
    
    # Create the plot folder if it doesn't exist
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Create subplots for each feature
    fig, axes = plt.subplots(num_features, 2, figsize=(10, 3 * num_features))

    for i, feature in enumerate(feature_columns):
        # Before SMOTE
        sns.histplot(original_df[feature], kde=True, ax=axes[i, 0], color='blue')
        axes[i, 0].set_title(f"Before SMOTE: {feature}")

        # After SMOTE
        sns.histplot(smote_df[feature], kde=True, ax=axes[i, 1], color='red')
        axes[i, 1].set_title(f"After SMOTE: {feature}")

    # Save the plot as an image in the plot folder after the loop
    plot_filename = os.path.join(plot_folder, "feature_distributions.png")
    plt.tight_layout()
    plt.savefig(plot_filename)  # Save the plot
    plt.close()  # Close the plot to release memory

    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    plot_feature_distributions()
