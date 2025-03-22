This is for ADL 


rmbr ur virtual env: venv\Scripts\Activate
then pip install requirements.txt

Instructions:
1. import in the original excel as raw_data.csv into folder "data"
2. run python src/data_preprocessing.py, this will create a clean data file
3. Run python src/smote.py, this creates synthetic data for the minority class (in our case its DR Present)
4. Run python src/data_comparision.py to get the scatter plots



Todos:
1. Train Test Validation Split on data_processing.py
1. Normalization for Train dataset 
1. NN Model Testing
    1. 
1. Set up function for Stratified K Fold, n_splits=5
1. Model Evaluation: Loss and Other Metrics