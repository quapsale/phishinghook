import os
import pickle
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read the CSV file directly
csv_path = 'opcodes.csv'
logging.info(f"Reading data from {csv_path}")
data = pd.read_csv(csv_path)

# Parse the data and create histograms
contract_groups = data.groupby('Contract')
histograms = []
labels = []
contracts = []

for contract, group in contract_groups:
    if contract == 'Contract':
        continue  # Skip the header row
    opcode_histogram = Counter(group['Opcode'])
    histograms.append(opcode_histogram)
    labels.append(group['Label'].iloc[0])
    contracts.append(contract)

# Convert histograms to feature vectors
all_opcodes = set(data['Opcode'])
feature_vectors = []
labels = np.array(labels)

for histogram in histograms:
    feature_vector = [histogram[opcode] for opcode in all_opcodes]
    feature_vectors.append(feature_vector)

# Convert feature vectors to a NumPy array
feature_vectors = np.array(feature_vectors)

# Save feature vectors and labels
feature_vectors_file = 'feature_vectors.pkl'
with open(feature_vectors_file, 'wb') as f:
    pickle.dump((feature_vectors, labels), f)

# Function to train and evaluate a model
def kfold_train_and_evaluate_model(model, model_name, num_folds, random_state, run_number):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    # List to store all fold results
    all_results = []

    for fold, (train_index, test_index) in enumerate(kf.split(feature_vectors)):
        X_train, X_test = feature_vectors[train_index], feature_vectors[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Train
        model.fit(X_train, y_train)

        # Test
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store results in a dictionary for the current fold
        fold_results = {
            'Run': run_number,
            'Model': model_name,
            'Fold': fold + 1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        all_results.append(fold_results)

        # Log fold results
        logging.info(f"{model_name} Run {run_number} Fold {fold + 1} Results: {fold_results}")

    # Save all results for the current run in a single CSV file
    results_dir = f"run_{run_number}"
    os.makedirs(results_dir, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, f"{model_name}_results.csv"), index=False)

    # Log overall run completion
    logging.info(f"{model_name} Run {run_number}: Results saved for all folds.")


# Load best hyperparameters from Optuna studies
def load_best_params(study_file):
    with open(study_file, 'rb') as f:
        study = pickle.load(f)
    return study.best_params

# Load best parameters for each model
best_params = {
    'RandomForest': load_best_params('optuna_study_RandomForest.pkl'),
    'KNN': load_best_params('optuna_study_KNN.pkl'),
    'LogisticRegression': load_best_params('optuna_study_LogisticRegression.pkl'),
    'SVM': load_best_params('optuna_study_SVM.pkl'),
    'XGBoost': load_best_params('optuna_study_XGBoost.pkl'),
    'CatBoost': load_best_params('optuna_study_CatBoost.pkl'),
    'LightGBM': load_best_params('optuna_study_LightGBM.pkl'),
}

num_folds = 10
seed = 42

# Define models with best parameters
models = {
    'RandomForest': RandomForestClassifier(**best_params['RandomForest'], random_state=seed),
    'KNN': KNeighborsClassifier(**best_params['KNN'],),
    'LogisticRegression': LogisticRegression(**best_params['LogisticRegression'], random_state=seed),
    'SVM': SVC(**best_params['SVM'], random_state=seed),
    'XGBoost': XGBClassifier(**best_params['XGBoost'], random_state=seed),
    'CatBoost': CatBoostClassifier(**best_params['CatBoost'], random_state=seed, verbose=0),
    'LightGBM': LGBMClassifier(**best_params['LightGBM'], random_state=seed)
}

# Train and evaluate each model for three runs
for run_number in range(1, 4):  # 1 to 3
    for model_name, model in models.items():
        logging.info(f"Run {run_number}: Training and evaluating {model_name}")
        kfold_train_and_evaluate_model(model, model_name, num_folds, seed, run_number)
