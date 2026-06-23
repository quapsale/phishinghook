import os
import pickle
import pandas as pd
import numpy as np
import logging
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset paths
data_dir = 'dataset/disassembled/'
train_csv = os.path.join(data_dir, 'train.csv')
test_files = [os.path.join(data_dir, f'test_set_{i}.csv') for i in range(1, 10)]

# Load training data
logging.info(f"Reading training data from {train_csv}")
train_data = pd.read_csv(train_csv)

# Parse training data and create histograms
train_groups = train_data.groupby('Contract')
train_histograms = []
train_labels = []

for contract, group in train_groups:
    opcode_histogram = Counter(group['Opcode'])
    train_histograms.append(opcode_histogram)
    train_labels.append(group['Label'].iloc[0])

# Convert histograms to feature vectors
all_opcodes = set(train_data['Opcode'])
train_feature_vectors = []

for histogram in train_histograms:
    feature_vector = [histogram[opcode] for opcode in all_opcodes]
    train_feature_vectors.append(feature_vector)

# Convert to NumPy arrays
train_feature_vectors = np.array(train_feature_vectors)
train_labels = np.array(train_labels)

# Load best hyperparameters
with open('optuna_study_RandomForest.pkl', 'rb') as f:
    best_params = pickle.load(f).best_params

# Initialize model
model = RandomForestClassifier(**best_params, random_state=42)

# Train model
logging.info("Training RandomForest model...")
train_start = time.time()
model.fit(train_feature_vectors, train_labels)
train_end = time.time()
logging.info(f"Training completed in {train_end - train_start:.2f} seconds")

# Evaluate on test sets
final_results = []

for i, test_file in enumerate(test_files, start=1):
    logging.info(f"Evaluating on {test_file}")
    test_data = pd.read_csv(test_file)
    timestamp = pd.to_datetime(test_data['Timestamp']).dt.to_period('M').unique()[0]
    
    # Parse test data
    test_groups = test_data.groupby('Contract')
    test_histograms = []
    test_labels = []
    
    for contract, group in test_groups:
        opcode_histogram = Counter(group['Opcode'])
        test_histograms.append(opcode_histogram)
        test_labels.append(group['Label'].iloc[0])
    
    # Convert to feature vectors
    test_feature_vectors = []
    
    for histogram in test_histograms:
        feature_vector = [histogram[opcode] for opcode in all_opcodes]
        test_feature_vectors.append(feature_vector)
    
    test_feature_vectors = np.array(test_feature_vectors)
    test_labels = np.array(test_labels)
    
    # Predict
    inference_start = time.time()
    y_pred = model.predict(test_feature_vectors)
    inference_end = time.time()
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, y_pred)
    precision_0 = precision_score(test_labels, y_pred, pos_label=0)
    recall_0 = recall_score(test_labels, y_pred, pos_label=0)
    f1_0 = f1_score(test_labels, y_pred, pos_label=0)
    precision_1 = precision_score(test_labels, y_pred, pos_label=1)
    recall_1 = recall_score(test_labels, y_pred, pos_label=1)
    f1_1 = f1_score(test_labels, y_pred, pos_label=1)
    
    # Store results
    final_results.append({
        'Test Split': f'test_set_{i}',
        'Timestamp': str(timestamp),
        'Accuracy': accuracy,
        'Benign Precision': precision_0,
        'Benign Recall': recall_0,
        'Benign F1': f1_0,
        'Phishing Precision': precision_1,
        'Phishing Recall': recall_1,
        'Phishing F1': f1_1
    })

# Save results
results_df = pd.DataFrame(final_results)
results_df.to_csv("results/random_forest.csv", index=False)
logging.info("All experiments completed and results saved.")
