import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class DataProcessor:
    def __init__(self, filename, lookback, batch_size=32):
        self.filename = filename
        self.lookback = lookback
        self.batch_size = batch_size

    @staticmethod
    def convert_string_to_float(s, i=None, j=None):

        # Check if string contains non-numeric characters (excluding comma and period)
        for char in s:
            if not (char.isdigit() or char in ',.'):
                raise ValueError(f"String contains non-numeric characters at row {i}, column {j}")

        # Count periods and commas in the string
        period_count = s.count('.')
        comma_count = s.count(',')

        # For 'Όγκος' column
        if period_count > 1 and comma_count == 0:
            # Remove periods (thousands separators) and convert to float
            s = s.replace('.', '')
            return float(s)
        # For 'Τζίρος' column
        elif period_count > 1 and comma_count == 1:
            # Remove periods (thousands separators), replace comma with period (decimal point) and convert to float
            s = s.replace('.', '').replace(',', '.')
            return float(s)
        # For the rest of the columns
        else:
            # Replace comma with period (decimal point) and convert to float
            s = s.replace(',', '.')
            return float(s)

    @staticmethod
    def process_dataframe(data):

        # Make a copy of the original dataframe to avoid modifying it directly
        data_processed = data.copy()

        # Identify rows with NaN values
        nan_rows = data_processed.isnull().any(axis=1)

        # Collect indices of rows with NaN values
        nan_indices = [i for i, is_nan in enumerate(nan_rows) if is_nan]

        # Print row indices and drop rows
        for i in nan_indices:
            print(f"Row {i} contains NaN. Deleting the row.")
        data_processed.drop(nan_indices, inplace=True)

        # Reset dataframe index after dropping rows
        data_processed.reset_index(drop=True, inplace=True)

        # Loop over rows
        for i, row in data_processed.iterrows():
            # Loop over columns
            for j, cell in enumerate(row):
                # Skip the first (index 0) and third (index 2) columns
                if j not in [0, 2]:
                    try:
                        # Try to convert the cell to a float
                        if pd.isna(cell):
                            print(f"NaN found at row {i}, column {j}")
                        else:
                            data_processed.iat[i, j] = convert_string_to_float(str(cell), i, j)
                    except ValueError as e:
                        # If a ValueError is raised, add information about the row and column
                        raise ValueError(f"Error in row {i}, column {j}: {e}")

        return data_processed

    @staticmethod
    def load_and_preprocess_data(filename, lookback, batch_size):
        # Read CSV and preprocess
        data = pd.read_csv(filename, skiprows=1)
        data = DataProcessor.process_dataframe(data)

        # Select and reorder columns
        data = data.iloc[:, [0, 1, 3, 4, 5, 6, 7]]
        data.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Turnover']
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
        data.sort_values(by='Date', ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Extract day, month, year as features
        data['Day'] = data['Date'].dt.day.astype(float)
        data['Month'] = data['Date'].dt.month.astype(float)
        data['Year'] = data['Date'].dt.year.astype(float)

        # Create sequences for training
        inputs = []
        targets = []
        for i in range(len(data) - lookback - 7):
            inputs.append(data.iloc[i:i + lookback, 1:].values)
            targets.append(
                data.iloc[i + lookback:i + lookback + 7, 1].values)  # Predicting 'Close' prices for next 7 days

        inputs, targets = np.array(inputs), np.array(targets)

        # Normalize the features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        inputs = np.array([scaler.fit_transform(x) for x in inputs])
        targets = scaler.transform(targets)

        # Convert to PyTorch tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Create datasets
        dataset = TensorDataset(inputs, targets)

        # Split the data: 70% train, 20% validation, 10% test
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                 [train_size, val_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        return scaler, train_loader, val_loader, test_loader, data['Close'].values


class CheckpointManager:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def save(self, epoch, model, optimizer, loss):
        checkpoint_path = os.path.join(self.folder_path, f"checkpoint_{epoch}.tar")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

    def load(self, checkpoint_path, model, optimizer):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss


class ModelManager:
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        self.model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.MSELoss()

    def train(self, train_dataset, val_dataset, checkpoint_manager=None, start_epoch=0, num_epochs=100):
        # TODO: Actual Training Loop Here
        pass

    def rolling_window_validation(self, data, initial_window, val_size, step_size, lookback):
        total_length = len(data)
        start_idx = 0

        while (start_idx + initial_window + val_size) <= total_length:
            train_window = data[start_idx:start_idx + initial_window]
            val_window = data[start_idx + initial_window:start_idx + initial_window + val_size]

            # Here, train the model on train_window and validate on val_window
            # Note: Both train_window and val_window should be divided into sequences based on lookback and prediction days (7 days)
            # TODO: Training and Validation on the windows

            start_idx += step_size


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Taking the last output to predict the next 7 days
        return out


def main():
    # Paths and Hyperparameters
    FILE_PATH = "your_file.csv"
    LOOKBACK = 30
    BATCH_SIZE = 32
    CHECKPOINT_FOLDER = "checkpoints"

    # Data Processing
    _, train_dataset, val_dataset, test_dataset, _ = DataProcessor.load_and_preprocess_data(FILE_PATH, LOOKBACK, BATCH_SIZE)

    # Checkpoint Management
    checkpoint_manager = CheckpointManager(folder_path=CHECKPOINT_FOLDER)

    # Model Training using Rolling Window Validation
    model_manager = ModelManager(input_dim=9, hidden_dim=50, num_layers=1, output_dim=7)  # Updated output_dim to predict for 7 days
    # Call the training function here

if __name__ == "__main__":
    main()