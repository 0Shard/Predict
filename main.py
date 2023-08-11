import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR



# Check if GPU is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a GPU and CUDA installed.")

# --- Data Preprocessing ---

class DataProcessor:

    @staticmethod
    def convert_string_to_float(s, i=None, j=None):
        for char in s:
            if not (char.isdigit() or char in ',.'):
                raise ValueError(f"String contains non-numeric characters at row {i}, column {j}")

        period_count = s.count('.')
        comma_count = s.count(',')

        if period_count > 1 and comma_count == 0:
            s = s.replace('.', '')
            return float(s)
        elif period_count > 1 and comma_count == 1:
            s = s.replace('.', '').replace(',', '.')
            return float(s)
        else:
            s = s.replace(',', '.')
            return float(s)

    @staticmethod
    def process_dataframe(data):
        data_processed = data.copy()
        nan_rows = data_processed.isnull().any(axis=1)
        nan_indices = [i for i, is_nan in enumerate(nan_rows) if is_nan]
        for i in nan_indices:
            print(f"Row {i} contains NaN. Deleting the row.")
        data_processed.drop(nan_indices, inplace=True)
        data_processed.reset_index(drop=True, inplace=True)

        for i, row in data_processed.iterrows():
            for j, cell in enumerate(row):
                if j not in [0, 2]:
                    try:
                        if pd.isna(cell):
                            print(f"NaN found at row {i}, column {j}")
                        else:
                            data_processed.iat[i, j] = DataProcessor.convert_string_to_float(str(cell), i, j)
                    except ValueError as e:
                        raise ValueError(f"Error in row {i}, column {j}: {e}")
        return data_processed

    @staticmethod
    def load_and_preprocess_data(filename, lookback, window_size, step_size, batch_size=32):
        data = pd.read_csv(filename, skiprows=1)
        data = DataProcessor.process_dataframe(data)
        data = data.iloc[:, [0, 1, 3, 4, 5, 6, 7]]
        data.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Turnover']
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
        data.sort_values(by='Date', ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data['Day'] = data['Date'].dt.day.astype(float)
        data['Month'] = data['Date'].dt.month.astype(float)
        data['Year'] = data['Date'].dt.year.astype(float)

        # Shift all columns except 'Close'
        for column in ['Open', 'High', 'Low', 'Volume', 'Turnover', 'Day', 'Month', 'Year']:
            data[f'Historical {column}'] = data[column].shift(lookback)

        data.drop(data.head(lookback).index, inplace=True)
        data.dropna(inplace=True)

        # Use a separate scaler for 'Close'
        scaler_close = MinMaxScaler(feature_range=(-1, 1))
        data['Close'] = scaler_close.fit_transform(data[['Close']])

        # Scaler for other columns
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data.iloc[:, 2:] = scaler.fit_transform(data.iloc[:, 2:])

        X, Y = [], []
        for i in range(len(data) - lookback - 7):
            X.append(data.iloc[i:i + lookback, 2:].values)  # All columns except 'Date' and 'Close'
            Y.append(data['Close'].values[i + lookback:i + lookback + 7])
        X, Y = np.array(X), np.array(Y)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        # Create PyTorch datasets and data loaders
        dataset = TensorDataset(X_tensor, Y_tensor)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size, val_size, test_size])

        # Rolling window for train and validation
        train_loaders, val_loaders = [], []
        for start_idx in range(0, len(train_dataset) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            train_subset = torch.utils.data.Subset(train_dataset, range(start_idx, end_idx))
            val_subset = torch.utils.data.Subset(val_dataset, range(start_idx, end_idx))
            train_loaders.append(DataLoader(train_subset, shuffle=False, batch_size=batch_size))
            val_loaders.append(DataLoader(val_subset, shuffle=False, batch_size=batch_size))

        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        return scaler, scaler_close, train_loaders, val_loaders, test_loader, data['Close'].values

# LSTM Model
class FinalCustomLSTMModelV2(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropouts, lookahead, num_features):
        super(FinalCustomLSTMModelV2, self).__init__()
        self.hidden_dims = hidden_dims
        self.lookahead = lookahead
        self.lstms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.lstms.append(nn.LSTM(input_dim, hidden_dims[0], batch_first=True))
        for i in range(1, len(hidden_dims)):
            self.dropouts.append(nn.Dropout(dropouts[i - 1]))
            self.lstms.append(nn.LSTM(hidden_dims[i - 1], hidden_dims[i], batch_first=True))
        self.fc = nn.Linear(hidden_dims[-1], lookahead)

    def forward(self, x):
        for i, lstm in enumerate(self.lstms):
            x, _ = lstm(x)
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        x = self.fc(x[:, -1, :])
        return x


def final_initialize_model_and_optimizer_v2(input_dim, hidden_dims, dropouts, lookahead, num_features):
    model = FinalCustomLSTMModelV2(input_dim=input_dim, hidden_dims=hidden_dims, dropouts=dropouts, lookahead=lookahead, num_features=num_features)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)  # Decrease LR every 10 epochs by a factor of 0.9
    return model, optimizer, scheduler

# Checkpoint Management
class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def save_checkpoint(self, epoch, model, optimizer, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"epoch_{epoch}.tar"))

    def load_checkpoint(self, checkpoint_file, model, optimizer):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

# Trainer
class ExtendedTrainer:
    def __init__(self, model, optimizer, scheduler, lookback, lookahead, window, step, checkpoint_manager):
        self.model = model
        self.optimizer = optimizer
        self.lookback = lookback
        self.lookahead = lookahead
        self.window = window
        self.scheduler = scheduler
        self.step = step
        self.checkpoint_manager = checkpoint_manager
        self.criterion = nn.MSELoss()
        self.training_loss = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for sequences_tensor, targets_tensor in train_loader:
            # Move tensors to GPU
            sequences_tensor = sequences_tensor.to('cuda:0')
            targets_tensor = targets_tensor.to('cuda:0')

            outputs = self.model(sequences_tensor)
            loss = self.criterion(outputs, targets_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.training_loss.append(avg_loss)

    def validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for sequences_tensor, targets_tensor in val_loader:
                sequences_tensor = sequences_tensor.to('cuda:0')
                targets_tensor = targets_tensor.to('cuda:0')
                outputs = self.model(sequences_tensor)
                loss = self.criterion(outputs, targets_tensor)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        return avg_val_loss

    def test(self, test_data):
        # Placeholder for now. You can expand this to evaluate the model on test data.
        pass

    def train(self, train_loaders, val_loaders, test_loader, num_epochs):
        start_epoch = 0
        continue_training = input("Would you like to continue training from a checkpoint? (yes/no): ").strip().lower()
        if continue_training == 'yes':
            checkpoint_file = input("Please provide the path to the checkpoint file: ")
            start_epoch, _ = self.checkpoint_manager.load_checkpoint(checkpoint_file, self.model, self.optimizer)

        for epoch in range(start_epoch, num_epochs):
            total_val_loss = 0.0
            # Use each rolling window for training and validation
            for train_loader, val_loader in zip(train_loaders, val_loaders):
                self.train_epoch(train_loader)
                val_loss = self.validate(val_loader)
                total_val_loss += val_loss

            avg_val_loss = total_val_loss / len(val_loaders)
            print(f"Epoch {epoch}/{num_epochs} - Avg Validation Loss: {avg_val_loss:.4f}")

            self.checkpoint_manager.save_checkpoint(epoch, self.model, self.optimizer, self.training_loss[-1])
            self.scheduler.step()  # Adjust the learning rate

        self.test(test_loader)

# Main Execution
csv_file_path = input("Please provide the path to the CSV file: ")
checkpoint_dir = input("Please provide the directory for saving checkpoints: ")

# Define lookback and batch_size
lookback = 30
batch_size = 32

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Load and preprocess data
scaler, scaler_close, train_loaders, val_loaders, test_loader, close_values = DataProcessor.load_and_preprocess_data(csv_file_path, lookback, 30, 7, batch_size)

# Initialize model and optimizer
model, optimizer, scheduler = final_initialize_model_and_optimizer_v2(9, [32, 64, 64], [0.2, 0.2], 7, 9)

# Move the model to GPU
model.to('cuda:0')

# Create checkpoint manager
checkpoint_manager = CheckpointManager(checkpoint_dir)

# Train the model
trainer = ExtendedTrainer(model, optimizer, scheduler, lookback, 7, 30, 7, checkpoint_manager)


try:
    trainer.train(train_loaders, val_loaders, test_loader, 100)  # Train for 100 epochs as an example
except RuntimeError as e:
    if "out of memory" in str(e):
        print("ERROR: GPU out of memory. Try reducing the batch size or model size.")
    else:
        raise e

# Save the model after training
torch.save(model.state_dict(), "final_model.pth")