import optuna
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CUDA device setup
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
configuration = GPT2Config.from_pretrained("gpt2", num_labels=1)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 requires explicit padding token

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
        
        # Ensure the label is returned as torch.long (integer type for CrossEntropyLoss)
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze(), torch.tensor(label, dtype=torch.float)

# Hyperparameters
num_folds = 10
seed = 42
max_seq_length = 128

def optuna_objective(trial):
    # Hyperparameter suggestions
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    patience = trial.suggest_int('patience', 5, 20)
    num_epochs = trial.suggest_int('epochs', 10, 50)

    # Prepare dataset
    dataset = OpcodeDataset(df, tokenizer, max_seq_length)

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

        # Model initialization
        model = GPT2ForSequenceClassification.from_pretrained("gpt2", config=configuration).to(device)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        model.config.pad_token_id = model.config.eos_token_id
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for input_ids, attention_mask, labels in train_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)  # Squeeze to match labels' shape
                loss = criterion(logits, labels)  # Compute BCE loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

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

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
            torch.cuda.empty_cache()

        # Evaluation on test set
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
        test_accuracies.append(accuracy)

    # Return mean accuracy across folds
    return np.mean(test_accuracies)

# Run Optuna
logging.info("Starting Optuna hyperparameter optimization")
study = optuna.create_study(direction='maximize')
study.optimize(optuna_objective, n_trials=5)

# Log the best parameters and results
logging.info(f"Best parameters: {study.best_params}")
logging.info(f"Best mean accuracy: {study.best_value}")

# Save the Optuna study for future reference
optuna_study_path = 'optuna_study_gpt2.pkl'
with open(optuna_study_path, "wb") as f:
    pickle.dump(study, f)
logging.info(f"Optuna study saved at {optuna_study_path}")
