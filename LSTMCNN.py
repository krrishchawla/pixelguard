import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Removed the fully connected layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

class LSTMCNN(nn.Module):
    def __init__(self, cnn, hidden_size, num_layers, num_classes, dropout_rate=0.4):
        super(LSTMCNN, self).__init__()
        self.cnn = cnn
        self.lstm = nn.LSTM(input_size=64 * 32 * 32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_out = []
        for t in range(timesteps):
            c_out_t = self.cnn(x[:, t, :, :, :])
            c_out.append(c_out_t)
        c_out = torch.stack(c_out, dim=1)

        r_out, _ = self.lstm(c_out)
        r_out2 = self.fc(r_out[:, -1, :])
        return r_out2


class ImageDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, sequence_length=10):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = {}
        directories = [(self.real_dir, 0), (self.fake_dir, 1)]

        for dir_path, label in directories:
            for file in sorted(os.listdir(dir_path)):
                video_id = file.split('_')[0]  # Assuming video ID is the first part of the filename
                if video_id not in samples:
                    samples[video_id] = {'images': [], 'label': label}
                samples[video_id]['images'].append(file)

        return samples

    def __len__(self):
        return sum(len(v['images']) for v in self.samples.values())

    def __getitem__(self, idx):
        flat_idx = 0
        for video_id, data in self.samples.items():
            num_frames = len(data['images'])
            if idx < flat_idx + num_frames:
                frame_idx = idx - flat_idx
                start_frame = max(frame_idx - self.sequence_length + 1, 0)
                sequence = data['images'][start_frame:frame_idx + 1]
                # If the sequence is shorter than sequence_length, repeat the last frame
                sequence += [data['images'][frame_idx]] * (self.sequence_length - len(sequence))
                break
            flat_idx += num_frames

        sequence_images = []
        for frame in sequence:
            image_path = os.path.join(self.real_dir if data['label'] == 0 else self.fake_dir, frame)
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                sequence_images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None, data['label']

        return torch.stack(sequence_images), torch.tensor(data['label'], dtype=torch.long)

    def split_dataset(self, test_split=0.3):
        video_ids = list(self.samples.keys())
        random.shuffle(video_ids)

        split_idx = int(len(video_ids) * (1 - test_split))
        train_video_ids = set(video_ids[:split_idx])
        test_video_ids = set(video_ids[split_idx:])

        train_samples = {vid: self.samples[vid] for vid in train_video_ids}
        test_samples = {vid: self.samples[vid] for vid in test_video_ids}

        return ImageDatasetSubset(self, train_samples), ImageDatasetSubset(self, test_samples)

class ImageDatasetSubset(Dataset):
    def __init__(self, parent_dataset, samples_subset):
        self.parent_dataset = parent_dataset
        self.samples = samples_subset
        self.transform = parent_dataset.transform
        self.sequence_length = parent_dataset.sequence_length

    def __len__(self):
        return sum(len(v['images']) for v in self.samples.values())

    def __getitem__(self, idx):
        flat_idx = 0
        for video_id, data in self.samples.items():
            num_frames = len(data['images'])
            if idx < flat_idx + num_frames:
                frame_idx = idx - flat_idx
                start_frame = max(frame_idx - self.sequence_length + 1, 0)
                sequence = data['images'][start_frame:frame_idx + 1]
                # If the sequence is shorter than sequence_length, repeat the last frame
                sequence += [data['images'][frame_idx]] * (self.sequence_length - len(sequence))
                break
            flat_idx += num_frames

        sequence_images = []
        for frame in sequence:
            image_path = os.path.join(self.parent_dataset.real_dir if data['label'] == 0 else self.parent_dataset.fake_dir, frame)
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                sequence_images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None, data['label']

        return torch.stack(sequence_images), torch.tensor(data['label'], dtype=torch.long)

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
full_dataset = ImageDataset(real_dir='./processed_faces_real', fake_dir='./processed_faces_fake', transform=transform)
train_dataset, test_dataset = full_dataset.split_dataset(test_split=0.3)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the LSTMCNN model
cnn_model = CNN()
lstmcnn_model = LSTMCNN(cnn_model, hidden_size=128, num_layers=5, num_classes=2, dropout_rate=0.4).to(device)

# Update the criterion and optimizer for the LSTMCNN model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstmcnn_model.parameters(), lr=0.005)

# Training Loop with tqdm Progress Bar
num_epochs = 10
train_loss = []
train_accuracy = []

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as tepoch:
        for sequences, labels in tepoch:
            if sequences is None:
                continue

            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = lstmcnn_model(sequences)
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

            # Update tqdm description for each batch
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# Save the model
torch.save(lstmcnn_model.state_dict(), 'lstmcnn_model_state_dict.pth')

# Plotting training loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
