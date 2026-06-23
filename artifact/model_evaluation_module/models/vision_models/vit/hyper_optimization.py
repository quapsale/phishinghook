import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from torchvision.models import vit_b_16
from math import ceil, isqrt
import torch.nn.functional as F
import logging
import pickle
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Load the CSV file
file_path = 'opcodes.csv'
df = pd.read_csv(file_path)

# Load lookup tables
with open('../utils/opcode_lookup.json', 'r') as f:
    opcode_lookup = json.load(f)

with open('../utils/operand_lookup.json', 'r') as f:
    operand_lookup = json.load(f)

with open('../utils/gas_lookup.json', 'r') as f:
    gas_lookup = json.load(f)

# Replace 'Name', 'Operand' and 'Gas' with corresponding keys from lookup tables
df['Opcode'] = df['Opcode'].map({v: float(k) for k, v in opcode_lookup.items()})
df['Operand'] = df['Operand'].map({v: float(k) for k, v in operand_lookup.items()})
df['Gas'] = df['Gas'].map({v: float(k) for k, v in gas_lookup.items()})

class OpcodeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __len__(self) -> int:
        return self.dataframe['Contract'].nunique()  # Number of unique contracts

    def __getitem__(self, idx: int):
        FILLER_NONE_VALUE = 0
        # Get the unique contract ID for this index
        contract_id = self.dataframe['Contract'].unique()[idx]
        # Filter the rows for the specific contract
        contract_rows = self.dataframe[self.dataframe['Contract'] == contract_id]

        # Initialize the 3D tensor with a filler value
        tensor = np.full((3, 224,224), FILLER_NONE_VALUE, dtype=float)

        # Fill the tensor with Opcode, Operand, and Gas values
        for i, (_, row) in enumerate(contract_rows.iterrows()):
            x, y = divmod(i, 224)
            if x >= 224 or y >= 224:
                continue  # Skip any indices out of bounds (shouldn't happen with the current logic)
            tensor[0, x, y] = row['Opcode']
            tensor[1, x, y] = row['Operand']
            tensor[2, x, y] = row['Gas']

        # Convert to tensor and add a batch dimension
        tensor = torch.tensor(tensor, dtype=torch.float32)  # shape (1, 3, square_size, square_size)

        # Fetch label from the specific `idx`
        label = torch.tensor(self.dataframe[self.dataframe['Contract'] == contract_id].iloc[0]['Label'], dtype=torch.float32)

        return tensor, label.view(-1)

logging.info(f'Building OpcodeDataset')
dataset = OpcodeDataset(df)
logging.info(f'OpcodeDataset built with {len(dataset)} samples')

# Hyperparameters
num_folds = 10
seed = 42

def optuna_objective(trial):
    # Hyperparameter suggestions
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    patience = trial.suggest_int('patience', 5, 20)
    num_epochs = trial.suggest_int('epochs', 10, 50)

    # K-fold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    test_accuracies = []

    for fold, (train_val_index, test_index) in enumerate(kf.split(dataset), 1):
        train_val_index, test_index = train_test_split(range(len(dataset)), test_size=0.2, random_state=seed)
        train_index, val_index = train_test_split(train_val_index, test_size=0.25, random_state=seed)

        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)
        test_dataset = Subset(dataset, test_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Initialize model, loss, and optimizer
        model = vit_b_16(weights='IMAGENET1K_V1').to(device)  # Load pre-trained ImageNet weights
        model.heads = nn.Linear(model.heads.head.in_features, 1).to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Training Average Loss: {train_loss:.4f}')
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_loss /= len(val_loader)
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Average Loss: {val_loss:.4f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logging.info(f'No improvement in validation loss for {epochs_without_improvement} epochs')
                if epochs_without_improvement >= patience:
                    logging.info(f'Early stopping triggered after {epoch+1} epochs.')
                    break
            torch.cuda.empty_cache()

        # Evaluation on test set
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                output_prob = torch.sigmoid(outputs)
                preds = (output_prob > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        test_accuracies.append(accuracy)

    # Return mean accuracy across folds
    return np.mean(test_accuracies)

# Run Optuna
logging.info("Starting Optuna hyperparameter optimization")
study = optuna.create_study(direction='maximize')
study.optimize(optuna_objective, n_trials=20)

# Log the best parameters and results
logging.info(f"Best parameters: {study.best_params}")
logging.info(f"Best mean accuracy: {study.best_value}")

# Save the Optuna study for future reference
optuna_study_path = 'optuna_study_vit.pkl'
with open(optuna_study_path, "wb") as f:
    pickle.dump(study, f)
logging.info(f"Optuna study saved at {optuna_study_path}")
