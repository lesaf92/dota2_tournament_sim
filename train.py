import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt

# --- Define the Neural Network (with updated input size) ---
class MatchPredictor(nn.Module):
    def __init__(self, num_features): # Now accepts the number of features
        super(MatchPredictor, self).__init__()
        # --- Input layer now accepts 'num_features' ---
        self.layer_1 = nn.Linear(num_features, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


class DotaDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_team_data(df):
    # --- Update team_data.json to save all ratings ---
    print("Generating team data for the app...")
    # This logic is more complex; for simplicity, we find the last row for each team
    last_matches = df.iloc[df.groupby('radiant_name')['match_id'].idxmax()]
    team_data = {}
    for _, row in last_matches.iterrows():
        team_data[row['radiant_name']] = {
            'elo32': row['radiant_elo32_before'],
            'elo64': row['radiant_elo64_before'],
            'glicko_mu': row['radiant_glicko_mu_before'],
            'glicko_rd': row['radiant_glicko_rd_before'],
        }
    # This is a simplified approach
    with open('team_data.json', 'w') as f:
        json.dump(team_data, f, indent=4)

def main():
    # --- Load and Prepare Data ---
    print("Loading multi-rating dataset...")
    df = pd.read_csv('dota2_multi_rating_dataset.csv') # Use the new dataset

    # --- KEY CHANGE: Engineer multiple features ---
    df['elo32_diff'] = df['radiant_elo32_before'] - df['dire_elo32_before']
    df['elo64_diff'] = df['radiant_elo64_before'] - df['dire_elo64_before']
    df['glicko_mu_diff'] = df['radiant_glicko_mu_before'] - df['dire_glicko_mu_before']
    # We can also include the raw Rating Deviations (RD) as features
    df['radiant_rd'] = df['radiant_glicko_rd_before']
    df['dire_rd'] = df['dire_glicko_rd_before']

    # Define which columns are our features
    feature_columns = ['elo32_diff', 'elo64_diff', 'glicko_mu_diff', 'radiant_rd', 'dire_rd']
    X = df[feature_columns].values
    y = df['radiant_win'].values.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    train_dataset = DotaDataset(X_train_tensor, y_train_tensor)
    val_dataset = DotaDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    # --- Initialize Model with the number of features ---
    num_features = len(feature_columns)
    model = MatchPredictor(num_features=num_features)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    epochs = 50 # Increased epochs to better see the trends
    print("\nStarting model training...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # --- Store average training loss for the epoch ---
        history['train_loss'].append(epoch_train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            epoch_val_loss = 0
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            # --- Store validation loss and accuracy for the epoch ---
            history['val_loss'].append(epoch_val_loss / len(val_loader))
            history['val_accuracy'].append(accuracy)

            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {history["train_loss"][-1]:.4f}, Val Loss: {history["val_loss"][-1]:.4f}, Val Accuracy: {accuracy:.2f}%')

    # --- Saving model ---
    print("\nTraining complete. Saving model and scaler...")
    # Set the model to evaluation mode
    model.eval()
    # Create a dummy input with the correct shape to trace the model
    dummy_input = torch.randn(1, num_features)
    # Trace the model to create a TorchScript module
    traced_model = torch.jit.trace(model, dummy_input)
    # Save the traced model in .pt format
    torch.jit.save(traced_model, 'dota2_predictor.pt')
    # --- Saving scaler ---
    with open('scaler.json', 'w') as f:
        json.dump({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}, f)
    # --- Saving team data ---
    create_team_data(df)

    # --- NEW: Plotting the results ---
    print("Generating training plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot Loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot Accuracy
    ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='green')
    ax2.set_title('Model Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_performance.png')
    print("Saved training performance graph to 'training_performance.png'")
    plt.close()



if __name__ == '__main__':
    main()
    print("Successfully saved 'dota2_predictor.pt', 'scaler.json', and 'team_data.json'.")
