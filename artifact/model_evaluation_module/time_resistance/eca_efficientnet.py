import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import models
import logging
import re
from datetime import datetime
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load best hyperparameters from Optuna studies
def load_best_params(study_file):
    with open(study_file, 'rb') as f:
        study = pickle.load(f)
    return study.best_params

best_params = load_best_params('optuna_study_eca_efficientnet.pkl')

# Define dataset class
class OpcodeDataset(Dataset):
    def __init__(self, df, target_height=224, target_width=224):
        self.df = df
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self):
        return len(self.df)

    def bytecode_to_image_tensor(self, bytecode):
        potential_colors = re.findall(r'[0-9a-fA-F]{6}', bytecode)
        rgb_colors = [(int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)) for color in potential_colors]
        target_pixels = self.target_height * self.target_width
        flat_rgb = np.array(rgb_colors, dtype=np.uint8).flatten()
        padded_flat_rgb = np.pad(flat_rgb, (0, 3 * target_pixels - flat_rgb.size), constant_values=0)
        reshaped_hwc = padded_flat_rgb.reshape((self.target_height, self.target_width, 3))
        image_tensor_chw = np.transpose(reshaped_hwc, (2, 0, 1))
        return image_tensor_chw

    def __getitem__(self, idx):
        bytecode = self.df.iloc[idx]['bytecode']
        label = self.df.iloc[idx]['label']
        image_tensor = self.bytecode_to_image_tensor(bytecode)
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image_tensor, label_tensor

# Define the model
class EfficientNetB0Modified(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetB0Modified, self).__init__()
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1').features
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(1280, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        gwap = self.global_avg_pool(x)
        output = self.fc(gwap)
        output = output.view(-1)
        return output

# Load training data
train_df = pd.read_csv("dataset/raw/train.csv")
logging.info(f"Training dataset loaded with {len(train_df)} samples")
train_dataset = OpcodeDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, pin_memory=True)

# Model setup
model = EfficientNetB0Modified(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

# Train the model
num_epochs = best_params['epochs']
logging.info("Starting training")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    logging.info(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
logging.info("Training complete")

# Evaluate on test sets
test_results = []
for i in range(1, 10):
    test_df = pd.read_csv(f"dataset/raw/test_set_{i}.csv")
    test_dataset = OpcodeDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, pin_memory=True)
    
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
results_df.to_csv("results/eca_efficientnet.csv", index=False)
logging.info("Testing complete. Results saved.")
