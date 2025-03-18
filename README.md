This is for ADL 

Instructions:
1. import in the original excel as raw_data.csv
2. run python src/data_preprocessing.py, this will create a clean data file
3. Run python src/smote.py, this creates synthetic data for the minority class (in our case its DR Present)
4. Run python src/data_comparision.py to get the scatter plots