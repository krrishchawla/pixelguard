import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CNN_Features(nn.Module):
    def __init__(self):
        super(CNN_Features, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the features
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN_Features()
        self.lstm = nn.LSTM(input_size=64 * 32 * 32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        batch_size, sequence_length, C, H, W = x.size()
        c_in = x.view(batch_size * sequence_length, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, sequence_length, -1)

        # Pack the sequence for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(r_in, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # We use the output of the last time step for classification
        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(len(lengths), output.size(2))
        idx = idx.unsqueeze(1).to(x.device)
        last_output = output.gather(1, idx).squeeze(1)

        return self.fc(last_output)


from collections import defaultdict
import random

class VideoFrameDataset(Dataset):
    def __init__(self, real_dir, fake_dir, max_sequence_length, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.max_sequence_length = max_sequence_length
        self.transform = transform
        self.samples = self._load_samples()

    def _parse_filename(self, filename):
        parts = filename.split('_')
        unique_id = '_'.join(parts[:-2])
        frame_num = int(parts[-2])
        return unique_id, frame_num

    def _load_samples(self):
        real_samples = defaultdict(list)
        fake_samples = defaultdict(list)

        for filename in sorted(os.listdir(self.real_dir)):
            unique_id, frame_num = self._parse_filename(filename)
            real_samples[unique_id].append((frame_num, filename))

        for filename in sorted(os.listdir(self.fake_dir)):
            unique_id, frame_num = self._parse_filename(filename)
            fake_samples[unique_id].append((frame_num, filename))

        samples = []
        for unique_id in real_samples:
            real_samples[unique_id].sort()  # Sort by frame number
            fake_samples[unique_id].sort()  # Sort by frame number

            num_sequences = min(len(real_samples[unique_id]), len(fake_samples[unique_id])) // self.max_sequence_length
            for seq_idx in range(num_sequences):
                start_idx = seq_idx * self.max_sequence_length
                end_idx = start_idx + self.max_sequence_length
                samples.append((unique_id, start_idx, end_idx))

        return samples

    def __getitem__(self, idx):
        unique_id, start_idx, end_idx = self.samples[idx]
        real_frames = [f[1] for f in real_samples[unique_id][start_idx:end_idx]]
        fake_frames = [f[1] for f in fake_samples[unique_id][start_idx:end_idx]]

        real_images = [self._load_image(self.real_dir, frame) for frame in real_frames]
        fake_images = [self._load_image(self.fake_dir, frame) for frame in fake_frames]

        # Padding if necessary
        real_images = self._pad_images(real_images)
        fake_images = self._pad_images(fake_images)

        return torch.stack(real_images), torch.stack(fake_images), len(real_images)

    def _load_image(self, dir_path, filename):
        image_path = os.path.join(dir_path, filename)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def _pad_images(self, images):
        num_frames = len(images)
        for _ in range(self.max_sequence_length - num_frames):
            images.append(torch.zeros_like(images[0]))  # Assuming images[0] has the correct shape
        return images

    def __len__(self):
        return len(self.samples)



from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize model, loss function, optimizer, etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM(num_classes=2, hidden_size=512, num_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dataset and DataLoader
max_sequence_length = 10  # Adjust as needed
train_dataset = VideoFrameDataset(real_dir='./processed_faces_real', 
                                  fake_dir='./processed_faces_fake', 
                                  max_sequence_length=max_sequence_length, 
                                  transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)


# Training loop with tqdm progress bar
num_epochs = 5  # Adjust as needed
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as tepoch:
        for sequences, labels, lengths in tepoch:
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update tqdm description for each batch
            tepoch.set_postfix(loss=(running_loss / (tepoch.n + 1)), accuracy=100. * correct / total)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
