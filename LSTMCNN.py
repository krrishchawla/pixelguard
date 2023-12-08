import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(CNNLSTM, self).__init__()
        # CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # LSTM
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(64 * 32 * 32, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.pool(F.relu(self.conv1(c_in)))
        c_out = self.pool(F.relu(self.conv2(c_out)))
        c_out = self.pool(F.relu(self.conv3(c_out)))
        c_out = c_out.view(batch_size, timesteps, -1)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(c_out.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(c_out.device)

        # Forward propagate LSTM
        out, _ = self.lstm(c_out, (h0, c0))
        out = out[:, -1, :]

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


class FrameSequenceDataset(Dataset):
    def __init__(self, real_dir, fake_dir, sequence_length=10, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        samples += self._create_samples_from_directory(self.real_dir, 0)  # Label 0 for real
        samples += self._create_samples_from_directory(self.fake_dir, 1)  # Label 1 for fake
        return samples

    def _create_samples_from_directory(self, directory, label):
        image_files = sorted(os.listdir(directory))
        grouped_files = {}

        # Group images by their unique ID
        for file in image_files:
            # Filename format: 'lips_UNIQUEid_someNumber_0'
            unique_id = file.split('_')[1]  # Adjust this based on your filename format
            if unique_id not in grouped_files:
                grouped_files[unique_id] = []
            grouped_files[unique_id].append(file)

        # Create samples
        samples = []
        for files in grouped_files.values():
            for i in range(0, len(files) - self.sequence_length + 1):
                sequence_files = files[i:i + self.sequence_length]
                samples.append({'images': sequence_files, 'label': label})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence = []
        for image_name in sample['images']:
            image_path = os.path.join(self.real_dir if sample['label'] == 0 else self.fake_dir, image_name)
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            sequence.append(image)
        sequence = torch.stack(sequence)
        return sequence, sample['label']

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and data loader
train_dataset = FrameSequenceDataset(real_dir='./extracted_lips_real', 
                                     fake_dir='./extracted_lips_fake', 
                                     transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(num_classes=2, hidden_size=128, num_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training the model
num_epochs = 10
train_loss = []
train_accuracy = []

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    for i, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# After training, save the model
torch.save(model.state_dict(), 'cnn_lstm_model_state_dict.pth')

# If you want to save the entire model (optional)
torch.save(model, 'cnn_lstm_model_entire.pth')

# Plotting training loss and accuracy
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# num_epochs = 10

# for epoch in range(num_epochs):
#     for i, (sequences, labels) in enumerate(train_loader):
#         sequences = sequences.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(sequences)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 100 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
