from imblearn.over_sampling import ADASYN
from imblearn.datasets import make_imbalance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os


plot_folder = "plots"

def apply_adasyn(input_file="data/processed_data.csv", output_file="data/processed_data_adasyn.csv"):
    df = pd.read_csv(input_file)
    X = df.drop(columns=['DR'])
    y = df['DR']
    X = X.applymap(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) and x.replace(',', '').replace('.', '').isdigit() else x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    adasyn = ADASYN(sampling_strategy='minority', n_neighbors=5, random_state=42)

    X_res, y_res = adasyn.fit_resample(X_train, y_train)

    print(f"Original dataset shape: {np.bincount(y_train)}")
    print(f"Resampled dataset shape: {np.bincount(y_res)}")

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    plt.figure(figsize=(10, 7))

    scatter1 = plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='coolwarm', marker='o', label='Original Data', alpha=0.6, s=50)

    scatter2 = plt.scatter(X_res.iloc[:, 0], X_res.iloc[:, 1], c=y_res, cmap='coolwarm', marker='x', label='Resampled Data', alpha=0.6, s=50)

    plt.title('Overlay of Original and Resampled Data (ADASYN)', fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)

    plt.legend(title="Data Type", fontsize=12)

    plt.colorbar(scatter1, ax=plt.gca(), label='Class Distribution')

    plot_filename = os.path.join(plot_folder, "adasyn_overlay_plot.png")
    plt.tight_layout()
    plt.savefig(plot_filename) 
    plt.close()  

    resampled_data = pd.DataFrame(X_res, columns=X.columns)
    resampled_data['DR'] = y_res
    resampled_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    apply_adasyn()
