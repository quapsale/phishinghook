import os
from tqdm import tqdm
import re
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# Save x and y tensors
tensor_file = 'tensor_data.pkl'
with open(tensor_file, 'wb') as f:
    pickle.dump((x, y, lengths), f)

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

best_params = load_best_params('optuna_study_scsguard.pkl')

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
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

        # Model initialization
        model = SCSGuard(vocab_size, embedding_dim, hidden_dim, num_heads).to(device)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        checkpoint_path = 'best_model_scsguard.pth'

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, (batch_x, batch_y, batch_lengths) in enumerate(train_loader):
                batch_x, batch_y, batch_lengths = batch_x.to(device), batch_y.to(device), batch_lengths.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x, batch_lengths)
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
                for batch_x, batch_y, batch_lengths in val_loader:
                    batch_x, batch_y, batch_lengths = batch_x.to(device), batch_y.to(device), batch_lengths.to(device)
                    outputs = model(batch_x, batch_lengths)
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
            for batch_x, batch_y, batch_lengths in test_loader:
                batch_x, batch_y, batch_lengths = batch_x.to(device), batch_y.to(device), batch_lengths.to(device)
                outputs = model(batch_x, batch_lengths)
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

# Hyperparameters
num_epochs = best_params['epochs']
num_folds = 10
seed = 42
model_name = 'SCSGuard'
max_seq_length = 128

# Build dataset
logging.info('Building OpcodeDataset')
dataset = OpcodeDataset(x, y, lengths)
logging.info(f'OpcodeDataset built with {len(dataset)} samples')

# Train and evaluate each model for three runs
for run_number in range(1, 4):  # 1 to 3
    logging.info(f"Run {run_number}: Training and evaluating {model_name}")
    kfold_train_and_evaluate_model(dataset, num_folds, num_epochs, seed, run_number)
