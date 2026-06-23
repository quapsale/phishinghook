import os
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from torchvision.models import vit_b_16
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import logging
import pickle

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
    
def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    return state['epoch'], state['val_loss']

best_params = {'learning_rate': 2.0119843798539834e-05, 'batch_size': 32, 'weight_decay': 2.4929055035669728e-05, 'patience': 10, 'epochs': 48}

def kfold_train_and_evaluate_model(dataset: OpcodeDataset, num_folds: int, num_epochs: int, seed: int, run_number: int):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # List to store all fold results
    all_results = []

    for fold, (train_val_index, test_index) in enumerate(kf.split(dataset), 1):
        # Split dataset into train, validation, and test sets
        train_val_index, test_index = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=seed
        )
        train_index, val_index = train_test_split(
            train_val_index, test_size=0.25, random_state=seed
        )  # 0.25 x 0.8 = 0.2

        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)
        test_dataset = Subset(dataset, test_index)

        # Create DataLoader for train, val, and test
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, pin_memory=True)

        # Initialize model, loss, and optimizer
        model = vit_b_16(weights='IMAGENET1K_V1').to(device)  # Load pre-trained ImageNet weights
        model.heads = nn.Linear(model.heads.head.in_features, 1).to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        checkpoint_path = 'best_model_vit_lookup.pth'

        # Training loop with validation
        for epoch in range(num_epochs):
            logging.info(f'Starting Epoch {epoch + 1}')
            model.train()
            epoch_loss = 0
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if (batch_idx + 1) % 10 == 0:
                    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Training Average Loss: {epoch_loss/len(train_loader):.4f}')

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
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Average Loss: {val_loss:.4f}')

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                logging.info(f'Validation loss improved to {val_loss:.4f}')
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            else:
                epochs_without_improvement += 1
                logging.info(f'No improvement in validation loss for {epochs_without_improvement} epochs')
                if epochs_without_improvement >= best_params['patience']:
                    logging.info(f'Early stopping triggered after {epoch+1} epochs.')
                    break
            torch.cuda.empty_cache()

        # Load the best model for inference
        load_checkpoint(checkpoint_path, model, optimizer)
        
        # Evaluation on test set
        logging.info('Evaluating on test set')
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                output_prob = torch.sigmoid(outputs)
                preds = (output_prob > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        # Store results in a dictionary for the current fold
        fold_results = {
            'Run': run_number,
            'Model': model_name,
            'Fold': fold,
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

logging.info(f'Building OpcodeDataset')
dataset = OpcodeDataset(df)
logging.info(f'OpcodeDataset built with {len(dataset)} samples')

# Save the dataset
dataset_file = 'opcode_dataset.pkl'
with open(dataset_file, 'wb') as f:
    pickle.dump(dataset, f)

# Hyperparameters
num_epochs = best_params['epochs']
num_folds = 10
seed = 42
model_name = 'ViT_lookup'

# Train and evaluate each model for three runs
for run_number in range(1, 4):  # 1 to 3
    logging.info(f"Run {run_number}: Training and evaluating {model_name}")
    kfold_train_and_evaluate_model(dataset, num_folds, num_epochs, seed, run_number)
