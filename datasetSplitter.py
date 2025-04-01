import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold



def datasetSplitterStratKFold(path:str=".\data\processed_data.csv",
                              dataFrame:pd.DataFrame = pd.DataFrame([]),
                              folds:int=5,
                              testSize = 0.1,
                              randomState:int = 42):
    """
    - Splits the dataset into trainingSets & testSets
    - Applies Stratified K Fold onto the training set

    Returns:
        [ (train_x, valid_x, train_y, valid_y) ] , X_test, y_test

        or (if folds = None)

        X_train, X_test, y_train, y_test
    """
    if not dataFrame.empty:
        data = dataFrame.copy()
    elif path:
        data = pd.read_csv(path)
    
    X = data.drop(columns=["DR"])
    y = pd.DataFrame(data["DR"])

    X_kfold, X_test, y_kfold, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState,stratify=y)

    if folds and folds > 1:
        kF = StratifiedKFold(n_splits=folds,shuffle=True, random_state=randomState)

        kFold = []

        for train_idx, test_idx in kF.split(X_kfold,y_kfold):
            # print(f"Fold: {fold}", train_idx, test_idx)
            # print(f"Frame: {fold}", X_kfold.shape, y_kfold.shape)
            train_x, train_y = X_kfold.iloc[train_idx], y_kfold.iloc[train_idx]
            test_x, test_y = X_kfold.iloc[test_idx], y_kfold.iloc[test_idx]
            # print(f"Fold: {fold}")
            # print(f"Train: {train_x.shape}, {train_y.shape}")
            # print(f"Test: {test_x.shape}, {test_y.shape}")
            kFold.append((train_x, test_x, train_y, test_y))

        return kFold, X_test, y_test
    else:
        return X_kfold, X_test, y_kfold, y_test
