from imblearn.over_sampling import ADASYN
from imblearn.datasets import make_imbalance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define the plot folder
plot_folder = "plots"

def apply_adasyn(input_file="data/processed_data.csv", output_file="data/processed_data_adasyn.csv"):
    df = pd.read_csv(input_file)
    # Initialize ADASYN with desired parameters
    X = df.drop(columns=['DR'])
    y = df['DR']
    X = X.applymap(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) and x.replace(',', '').replace('.', '').isdigit() else x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    adasyn = ADASYN(sampling_strategy='minority', n_neighbors=5, random_state=42)

    # Apply ADASYN to the training data
    X_res, y_res = adasyn.fit_resample(X_train, y_train)

    # Check the number of samples in each class after resampling
    print(f"Original dataset shape: {np.bincount(y_train)}")
    print(f"Resampled dataset shape: {np.bincount(y_res)}")

    # Create the plot folder if it doesn't exist
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Plot the original and resampled data on the same plot
    plt.figure(figsize=(10, 7))

    # Plot original data (convert X_train to numpy array for proper indexing)
    scatter1 = plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='coolwarm', marker='o', label='Original Data', alpha=0.6, s=50)

    # Plot resampled data (use iloc for pandas DataFrame indexing)
    scatter2 = plt.scatter(X_res.iloc[:, 0], X_res.iloc[:, 1], c=y_res, cmap='coolwarm', marker='x', label='Resampled Data', alpha=0.6, s=50)

    # Add titles and labels
    plt.title('Overlay of Original and Resampled Data (ADASYN)', fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)

    # Add a legend
    plt.legend(title="Data Type", fontsize=12)

    # Add color bar for class distribution
    plt.colorbar(scatter1, ax=plt.gca(), label='Class Distribution')

    # Save the plot as an image
    plot_filename = os.path.join(plot_folder, "adasyn_overlay_plot.png")
    plt.tight_layout()
    plt.savefig(plot_filename)  # Save the plot
    plt.close()  # Close the plot to release memory

    # Optionally save the resampled data to a CSV file
    resampled_data = pd.DataFrame(X_res, columns=X.columns)
    resampled_data['DR'] = y_res
    resampled_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    apply_adasyn()
