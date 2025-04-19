import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold



def datasetSplitterStratKFold(path:str=".\data\processed_data.csv",
                              dataFrame:pd.DataFrame = pd.DataFrame([]),
                              folds:int=5,
                              testSize = 0.1,
                              randomState:int = 42):
    """
    - Splits the dataset into trainingSets & testSets

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
    return X_kfold, X_test, y_kfold, y_test

def datasetFoldMaker(x_kFold:pd.DataFrame,
                     y_kFold:pd.DataFrame,
                     folds:int = 5,
                     normalize:bool=True,
                     randomState:int=42):
    """
    - Applies Stratified K Fold onto the training set
    - Normalizes X for each Fold

    Returns:
        [ (train_x, valid_x, train_y, valid_y) ]

        """
    if folds and folds > 1:
        kF = StratifiedKFold(n_splits=folds,shuffle=True, random_state=randomState)

        kFolds = []

        for train_idx, test_idx in kF.split(x_kFold,y_kFold):

            train_x, train_y = x_kFold.iloc[train_idx], y_kFold.iloc[train_idx]
            test_x, test_y = x_kFold.iloc[test_idx], y_kFold.iloc[test_idx]

            if normalize:
                train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
                test_x = (test_x - test_x.min()) / (test_x.max() - test_x.min())
            
            kFolds.append((train_x, test_x, train_y, test_y))

        return kFolds
    return None
