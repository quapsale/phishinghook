import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import T5Tokenizer, T5ForSequenceClassification, T5Config
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
import logging
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


logging.info(f'Loading dataset')
benign_hex = np.loadtxt('/path/to/benign/bytecodes.txt', dtype=str)
phishing_hex = np.loadtxt('/path/to/phishing/bytecodes.txt', dtype=str)
y_benign = [0] * len(benign_hex)
y_phishing = [1] * len(phishing_hex)

# Combine the data
names = np.concatenate((benign_hex, phishing_hex))  # Combine the Name column
labels = y_benign + y_phishing  # Combine the Label column

# Create a DataFrame
df = pd.DataFrame({
    'Label': labels,
    'Name': names
})
logging.info(f'Dataset loaded with {len(df)} samples')

# Initialize tokenizer and model
configuration = T5Config.from_pretrained("google-t5/t5-small", num_labels=1)
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

# Dataset class
class OpcodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = row['Label']
        sequence = row['Name']
        tokens = self.tokenizer(sequence, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze(), torch.tensor(label, dtype=torch.float)

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

# Load best hyperparameters from Optuna studies
def load_best_params(study_file):
    with open(study_file, 'rb') as f:
        study = pickle.load(f)
    return study.best_params

best_params = load_best_params('optuna_study_t5.pkl')

def kfold_train_and_evaluate_model(dataset: OpcodeDataset, num_folds: int, num_epochs: int, seed: int, run_number: int):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # List to store all fold results
    all_results = []

    for fold, (train_val_index, test_index) in enumerate(kf.split(dataset), 1):
        train_val_index, test_index = train_test_split(range(len(dataset)), test_size=0.2, random_state=seed)
        train_index, val_index = train_test_split(train_val_index, test_size=0.25, random_state=seed)
        train_index, val_index = train_test_split(train_val_index, test_size=0.25, random_state=seed)  # 0.25 x 0.8 = 0.2

        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)
        test_dataset = Subset(dataset, test_index)

        # Create DataLoader for train, val, and test
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, pin_memory=True)

        # Initialize model, loss, and optimizer
        model = T5ForSequenceClassification.from_pretrained("google-t5/t5-small", config=configuration).to(device)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        checkpoint_path = 'best_model_t5.pth'

        # Training loop with validation
        for epoch in range(num_epochs):
            logging.info(f'Starting Epoch {epoch + 1}')
            model.train()
            epoch_loss = 0
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)  # Squeeze to match labels' shape
                loss = criterion(logits, labels)  # Compute BCE loss
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
                for input_ids, attention_mask, labels in val_loader:
                    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.squeeze(-1)
                    loss = criterion(logits, labels)
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
            for input_ids, attention_mask, labels in test_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                probs = torch.sigmoid(logits)  # Convert logits to probabilities
                preds = (probs > 0.5).long()  # Threshold at 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
 
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

# Hyperparameters
num_epochs = best_params['epochs']
num_folds = 10
seed = 42
model_name = 'T5'
max_seq_length = 128

logging.info(f'Building OpcodeDataset')
dataset = OpcodeDataset(df, tokenizer, max_seq_length)
logging.info(f'OpcodeDataset built with {len(dataset)} samples')

# Train and evaluate each model for three runs
for run_number in range(1, 4):  # 1 to 3
    logging.info(f"Run {run_number}: Training and evaluating {model_name}")
    kfold_train_and_evaluate_model(dataset, num_folds, num_epochs, seed, run_number)
