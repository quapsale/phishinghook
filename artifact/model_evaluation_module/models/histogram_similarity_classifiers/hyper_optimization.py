import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from collections import Counter
import optuna

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

# Hyperparameters
max_iter = 50000
num_folds = 10
seed = 42

# Define the models
models = {
    'RandomForest': RandomForestClassifier,
    'KNN': KNeighborsClassifier,
    'LogisticRegression': LogisticRegression,
    'SVM': SVC,
    'XGBoost': XGBClassifier,
    'CatBoost': CatBoostClassifier,
    'LightGBM': LGBMClassifier
}

# Objective function for Optuna
def objective(trial, model_name):
    # Define model-specific hyperparameter ranges
    if model_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 2, 50)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    elif model_name == 'KNN':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 20)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_name == 'LogisticRegression':
        c = trial.suggest_loguniform('C', 1e-4, 1e2)
        model = LogisticRegression(C=c, solver='lbfgs', max_iter=max_iter, random_state=seed)
    elif model_name == 'SVM':
        c = trial.suggest_loguniform('C', 1e-4, 1e2)
        gamma = trial.suggest_loguniform('gamma', 1e-4, 1e-1)
        model = SVC(C=c, gamma=gamma, kernel='rbf', random_state=seed)
    elif model_name == 'XGBoost':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.5)
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=seed)
    elif model_name == 'CatBoost':
        iterations = trial.suggest_int('iterations', 50, 200)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.5)
        model = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, random_state=seed, verbose=0)
    elif model_name == 'LightGBM':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.5)
        model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Perform cross-validation
    scores = cross_val_score(model, feature_vectors, labels, cv=num_folds, scoring='accuracy')
    mean_score = scores.mean()

    # Optuna maximizes the objective function
    return mean_score

# Run hyperparameter optimization for each model
for model_name in models.keys():
    logging.info(f"Starting hyperparameter optimization for {model_name}")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_name), n_trials=20)

    # Get the best hyperparameters
    logging.info(f"Best hyperparameters for {model_name}: {study.best_params}")
    logging.info(f"Best accuracy for {model_name}: {study.best_value}")

    # Save the study results
    study_results_path = f"optuna_study_{model_name}.pkl"
    with open(study_results_path, "wb") as f:
        pickle.dump(study, f)
    logging.info(f"Saved Optuna study for {model_name} at {study_results_path}")
