import re
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load best hyperparameters from Optuna studies
def load_best_params(study_file):
    with open(study_file, 'rb') as f:
        study = pickle.load(f)
    return study.best_params

best_params = load_best_params('optuna_study_scsguard.pkl')

# Model Definition
class SCSGuard(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads):
        super(SCSGuard, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, lengths):
        x = self.embedding(x)
        attended_embeddings, _ = self.attn(x, x, x, need_weights=False)
        lengths_sorted, sorted_idx = torch.sort(lengths, descending=True)
        attended_sorted = attended_embeddings[sorted_idx]
        packed_input = pack_padded_sequence(attended_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, _ = self.gru(packed_input)
        gru_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, original_idx = torch.sort(sorted_idx)
        gru_output = gru_output[original_idx]
        lengths = lengths[original_idx]
        final_output = torch.stack([gru_output[i, lengths[i] - 1, :] for i in range(len(lengths))])
        final_output = self.dropout(final_output)
        logits = self.fc(final_output).squeeze()
        return logits

# Dataset Class
class OpcodeDataset(Dataset):
    def __init__(self, x, y, lengths):
        self.x = x
        self.y = y
        self.lengths = lengths

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.lengths[idx]

# Data Processing Functions
def hex_to_ngrams(hex_string):
    hex_string = hex_string[2:] if hex_string.startswith("0x") else hex_string
    return re.findall('......', hex_string)

def pad_sequences(sequences):
    max_length = max(len(seq) for seq in sequences)
    return [seq + [0] * (max_length - len(seq)) for seq in sequences]

# Load and Process Training Data
train_df = pd.read_csv("dataset/raw/train.csv")
train_df['Name'] = train_df['bytecode'].apply(hex_to_ngrams)
flat_bigrams = [bigram for sublist in train_df['Name'] for bigram in sublist]
codes, uniques = pd.factorize(np.array(flat_bigrams))
bigram_to_code = dict(zip(uniques, range(len(uniques))))
train_df['Name'] = train_df['Name'].apply(lambda bigram_list: [bigram_to_code[bigram] for bigram in bigram_list])
padded_bigrams = pad_sequences(train_df['Name'].tolist())
vocab_size = len(set(codes))
sequence_lengths = [len(seq) for seq in padded_bigrams]
x_train = torch.tensor(padded_bigrams, dtype=torch.long).to(device)
y_train = torch.tensor(train_df['label'].tolist(), dtype=torch.float32).to(device)
lens_train = torch.tensor(sequence_lengths, dtype=torch.long).to(device)

# Train Model
train_dataset = OpcodeDataset(x_train, y_train, lens_train)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
model = SCSGuard(vocab_size, 32, 100, 4).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
logging.info("Starting training")
num_epochs = best_params['epochs']
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y, batch_lengths in train_loader:
        batch_x, batch_y, batch_lengths = batch_x.to(device), batch_y.to(device), batch_lengths.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x, batch_lengths)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    logging.info(f"Epoch {epoch+1}: Training Loss: {epoch_loss / len(train_loader):.4f}")
logging.info("Training complete")

# Evaluate Model on Test Splits
test_results = []
for i in range(1, 10):
    test_df = pd.read_csv(f"dataset/raw/test_set_{i}.csv")
    test_month = test_df["creation_timestamp"].iloc[0][:7]
    test_df['Name'] = test_df['bytecode'].apply(hex_to_ngrams)
    test_df['Name'] = test_df['Name'].apply(lambda bigram_list: [bigram_to_code.get(bigram, 0) for bigram in bigram_list])
    padded_bigrams = pad_sequences(test_df['Name'].tolist())
    sequence_lengths = [len(seq) for seq in padded_bigrams]
    x_test = torch.tensor(padded_bigrams, dtype=torch.long).to(device)
    y_test = torch.tensor(test_df['label'].tolist(), dtype=torch.float32).to(device)
    lens_test = torch.tensor(sequence_lengths, dtype=torch.long).to(device)
    
    test_dataset = OpcodeDataset(x_test, y_test, lens_test)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_lengths in test_loader:
            batch_x, batch_y, batch_lengths = batch_x.to(device), batch_y.to(device), batch_lengths.to(device)
            outputs = model(batch_x, batch_lengths)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision_0 = precision_score(all_labels, all_preds, pos_label=0)
    recall_0 = recall_score(all_labels, all_preds, pos_label=0)
    f1_0 = f1_score(all_labels, all_preds, pos_label=0)
    
    precision_1 = precision_score(all_labels, all_preds, pos_label=1)
    recall_1 = recall_score(all_labels, all_preds, pos_label=1)
    f1_1 = f1_score(all_labels, all_preds, pos_label=1)
    
    timestamp = datetime.strptime(test_df.iloc[0]['creation_timestamp'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m")
    
    test_results.append({
        'Test Split': f'test_set_{i}',
        'Month': timestamp,
        'Accuracy': accuracy,
        'Benign Precision': precision_0,
        'Benign Recall': recall_0,
        'Benign F1': f1_0,
        'Phishing Precision': precision_1,
        'Phishing Recall': recall_1,
        'Phishing F1': f1_1
    })
    
    logging.info(f"Completed evaluation on test_set_{i}")

# Save results
results_df = pd.DataFrame(test_results)
results_df.to_csv("results/scsguard.csv", index=False)
logging.info("Testing complete. Results saved.")