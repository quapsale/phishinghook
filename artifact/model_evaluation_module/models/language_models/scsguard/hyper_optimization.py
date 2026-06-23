import optuna
from tqdm import tqdm
import re
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


class SCSGuard(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads):
        super(SCSGuard, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, lengths):
        # Embedding: Convert input bigram indices to embeddings
        x = self.embedding(x)

        # Apply Multi-head Attention
        attended_embeddings, _ = self.attn(x, x, x, need_weights=False)

        # Sort sequences by lengths (required for packing)
        lengths_sorted, sorted_idx = torch.sort(lengths, descending=True)
        attended_sorted = attended_embeddings[sorted_idx]

        # Pack padded sequence
        packed_input = pack_padded_sequence(attended_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)

        # GRU Forward Pass
        packed_output, _ = self.gru(packed_input)

        # Unpack sequences
        gru_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Restore original order
        _, original_idx = torch.sort(sorted_idx)
        gru_output = gru_output[original_idx]
        lengths = lengths[original_idx]

        # Extract last valid hidden state for each sequence
        final_output = torch.stack([gru_output[i, lengths[i] - 1, :] for i in range(len(lengths))])

        # Apply Dropout and Fully Connected Layer
        final_output = self.dropout(final_output)
        logits = self.fc(final_output).squeeze()

        return logits
    
def hex_to_ngrams(hex_string):
    # Remove the "0x" prefix if present
    hex_string = hex_string[2:] if hex_string.startswith("0x") else hex_string
    # Find bigrams in the hex string
    return re.findall('......', hex_string)

# Function to pad sequences to the size of bigram with max length
def pad_sequences(sequences):
    max_length = max(len(seq) for seq in sequences)  # Find the maximum length
    padded = [seq + [0] * (max_length - len(seq)) for seq in sequences]  # Pad with zeros
    return padded

# Convert each hex string in the 'Name' column to bigrams
df['Name'] = df['Name'].apply(hex_to_ngrams)

# Flatten the list of bigrams and factorize
flat_bigrams = [bigram for sublist in df['Name'] for bigram in sublist]

# Factorize the flat list of bigrams
codes, uniques = pd.factorize(np.array(flat_bigrams))

# Map the original bigrams back to their factorized codes
bigram_to_code = dict(zip(uniques, range(len(uniques))))

# Replace the bigram lists in the DataFrame with their factorized codes
df['Name'] = df['Name'].apply(lambda bigram_list: [bigram_to_code[bigram] for bigram in bigram_list])

# Pad the sequences in the 'Name' column
padded_bigrams = pad_sequences(df['Name'].tolist())

# Update the dataset creation to include sequence lengths
sequence_lengths = [len(seq) for seq in padded_bigrams]

# Convert to tensors
x = torch.tensor(padded_bigrams, dtype=torch.long).to(device)
y = torch.tensor(df['Label'].tolist(), dtype=torch.float32).to(device)
lengths = torch.tensor(sequence_lengths, dtype=torch.long).to(device)


class OpcodeDataset(Dataset):
    def __init__(self, x, y, lengths):
        self.x = x
        self.y = y
        self.lengths = lengths

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.lengths[idx]
    
# Hyperparameters
num_folds = 10
seed = 42
max_seq_length = 64
hidden_dim = 100
num_heads = 4
embedding_dim = 32

unique_bigrams = set()
for sublist in tqdm(padded_bigrams, desc="Extracting unique bigrams"): # This takes ~5 minutes in debug mode
    for bigram in sublist:
        unique_bigrams.add(bigram)
vocab_size = len(unique_bigrams)

# Build dataset
logging.info('Building OpcodeDataset')
dataset = OpcodeDataset(x, y, lengths)
logging.info(f'OpcodeDataset built with {len(dataset)} samples')

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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model initialization
        model = SCSGuard(vocab_size, embedding_dim, hidden_dim, num_heads).to(device)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y, batch_lengths in train_loader:
                batch_x, batch_y, batch_lengths = batch_x.to(device), batch_y.to(device), batch_lengths.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x, batch_lengths)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y, batch_lengths in val_loader:
                    batch_x, batch_y, batch_lengths = batch_x.to(device), batch_y.to(device), batch_lengths.to(device)
                    outputs = model(batch_x, batch_lengths)
                    loss = criterion(outputs, batch_y)
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
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_lengths in test_loader:
                batch_x, batch_y, batch_lengths = batch_x.to(device), batch_y.to(device), batch_lengths.to(device)
                outputs = model(batch_x, batch_lengths)
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
study.optimize(optuna_objective, n_trials=5)

# Log the best parameters and results
logging.info(f"Best parameters: {study.best_params}")
logging.info(f"Best mean accuracy: {study.best_value}")

# Save the Optuna study for future reference
optuna_study_path = 'optuna_study_scsguard.pkl'
with open(optuna_study_path, "wb") as f:
    pickle.dump(study, f)
logging.info(f"Optuna study saved at {optuna_study_path}")
