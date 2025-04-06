This is for ADL 


rmbr ur virtual env: venv\Scripts\Activate
then pip install requirements.txt

Instructions:
1. import in the original excel as raw_data.csv into folder "data"
2. run python src/data_preprocessing.py, this will create a clean data file
3. Run python src/smote.py, this creates synthetic data for the minority class (in our case its DR Present)
4. Run python src/data_comparision.py to get the scatter plots

Basic Model Workflow.ipynb
Read through this file to understand the step by step process of 
1. Making Folds
2. Preprocessing/Augmentation
3. Initialising Model
4. Training loop
5. Evaluating for each fold
6. Getting metrics of model
Disclaimer: This does not do cross-validation when training(which i overlooked but i too laze to code it in, Optuna2 train_and_evaluate(model,...) does it properly)

Optuna2.ipynb
regarding the training and cross-validation implementation, it has been fixed here. We track training loss against cv loss at each epoch, patience is how many epochs before early stopping. 
run requirements.txt again to install new libraries into your venv

This notebook is about trying to maximise a certain score
I made the score be the average of all 5 metrics
    1. Accuracy
    2. precision
    3. recall
    4. f1
    5. AUC
We evaluate the final average score across 5 folds, the folds loop is in the objective function maximise_combined_score(trial)
You can try different hyperparameter space and add or remove any hyperparams you dont want to optimise
IMPT: if last layer of your model is sigmoid, then don't use BCEwithlogitsloss, I have a failsafe in there that checks but optuna will still try and use it if you mention it in the hyperparameter space.
edit the epochs and patience within the objective study
edit the model definition/function so that you can test out different kinds of setups, and edit it in such a way that you may let optuna try values in certain aspects
when the thing is training you can open a dashboard at the last cell, it will prompt you, but then need to interrupt the code to stop it even after it has finished the trial.

