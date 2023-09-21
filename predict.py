import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

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
    def load_and_preprocess_data(filename, lookback, window_size, step_size, batch_size=35):
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

        # Use a separate scaler for 'Close'
        scaler_close = MinMaxScaler(feature_range=(-1, 1))
        data['Close'] = scaler_close.fit_transform(data[['Close']])

        # Scaler for other columns
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data.iloc[:, 2:] = scaler.fit_transform(data.iloc[:, 2:])

        # Ensure the dataset size is greater than 5000 points
        if len(data) <= 5000:
            raise ValueError("Dataset size should be greater than 5000 data points")

        # Create the X, Y arrays
        X, Y = [], []
        for i in range(len(data) - lookback - 7):  # 7 days lookahead for target
            X.append(data.iloc[i:i + lookback, 1:].values)  # Using all columns starting from 'Close' for training data
            Y.append(data['Close'].values[i + lookback:i + lookback + 7])  # Future 7 Close values as target
        X, Y = np.array(X), np.array(Y)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        # Create PyTorch datasets and data loaders
        dataset = TensorDataset(X_tensor, Y_tensor)
        train_size = int(0.6 * len(dataset))
        val_size = int(0.3 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_end = train_size
        val_end = train_end + val_size

        train_dataset = torch.utils.data.Subset(dataset, range(0, train_end))
        val_dataset = torch.utils.data.Subset(dataset, range(train_end, val_end))
        test_dataset = torch.utils.data.Subset(dataset, range(val_end, val_end + test_size))

        # Rolling window for train
        train_loaders = []
        for start_idx in range(0, len(train_dataset) - window_size + 1, step_size):
            train_end_idx = start_idx + window_size
            train_subset = torch.utils.data.Subset(train_dataset, range(start_idx, train_end_idx))
            train_loaders.append(DataLoader(train_subset, shuffle=False, batch_size=batch_size))

        # Adjust rolling window logic for validation
        val_loaders = []
        for start_idx in range(0, len(val_dataset) - window_size + 1, step_size):
            val_end_idx = start_idx + window_size
            val_subset = torch.utils.data.Subset(val_dataset, range(start_idx, val_end_idx))
            val_loaders.append(DataLoader(val_subset, shuffle=False, batch_size=batch_size))

        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        return scaler, scaler_close, train_loaders, val_loaders, test_loader, data['Close'].values


# LSTM Model
class FinalCustomLSTMModelV2(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropouts, lookahead):
        super(FinalCustomLSTMModelV2, self).__init__()
        self.hidden_dims = hidden_dims
        self.lookahead = lookahead
        self.lstm = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First LSTM layer
        self.lstm.append(nn.LSTM(input_dim, hidden_dims[0], batch_first=True))
        for i in range(1, len(hidden_dims)):
            self.dropouts.append(nn.Dropout(dropouts[i - 1]))
            self.lstm.append(nn.LSTM(hidden_dims[i - 1], hidden_dims[i], batch_first=True))

        # Fully connected layer
        self.fc = nn.Linear(hidden_dims[-1], lookahead)
        self.tanh = nn.Tanh()  # Activation function

    def forward(self, x):
        for i, lstm in enumerate(self.lstm):
            x, _ = lstm(x)
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        x = self.fc(x[:, -1, :])  # Take the last output from the LSTM sequence
        x = self.tanh(x)  # Apply the tanh activation function
        return x


# Load the trained model
model_path = input("Please provide the path to the trained model: ")
model = FinalCustomLSTMModelV2(input_dim=9, hidden_dims=[9, 9, 9, 9, 9, 9, 9], dropouts=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], lookahead=7)
model.load_state_dict(torch.load(model_path))
model.to('cuda:0')
model.eval()

# Load and preprocess input data
csv_file_path = input("Please provide the path to the CSV file for prediction: ")

# Define lookback
lookback = 28

# Load and preprocess data for prediction using DataProcessor from the first part
scaler, scaler_close, _, _, _, close_values = DataProcessor.load_and_preprocess_data(csv_file_path, lookback, 28, 7)

# Using the last lookback days for prediction
input_data = close_values[-lookback:]  # Take the last 'lookback' days
input_tensor = torch.tensor(input_data.reshape(1, lookback, -1), dtype=torch.float32).to('cuda:0')

# Predict future close prices
with torch.no_grad():
    predicted = model(input_tensor)

# Reverse the scaling transformation
predicted_close = scaler_close.inverse_transform(predicted.cpu().numpy().reshape(-1, 1))

print("Predicted future close prices:", predicted_close)

# Optional: Save the predictions to a CSV file
save_to_csv = input("Would you like to save the predictions to a CSV file? (yes/no): ").strip().lower()
if save_to_csv == 'yes':
    output_path = input("Please provide the path where you'd like to save the predictions: ")
    pd.DataFrame(predicted_close, columns=["Predicted Close"]).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
