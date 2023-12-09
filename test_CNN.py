import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(64 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Image Dataset Class
class ImageDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, start_idx=0, end_idx=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        self.samples = self._load_samples(start_idx, end_idx)

    def _load_samples(self, start_idx, end_idx):
        samples = []
        real_images = sorted(os.listdir(self.real_dir))[start_idx:end_idx]
        fake_images = sorted(os.listdir(self.fake_dir))[start_idx:end_idx]

        for file in tqdm(real_images, desc='Loading real images'):
            samples.append({'image': file, 'label': 0, 'dir': self.real_dir})

        for file in tqdm(fake_images, desc='Loading fake images'):
            samples.append({'image': file, 'label': 1, 'dir': self.fake_dir})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_name = sample['image']
        image_dir = sample['dir']
        image_path = os.path.join(image_dir, image_name)

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, sample['label']

        return image, sample['label']

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test Transforms (assuming same as training)
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test Dataset and DataLoader
test_dataset = ImageDataset(real_dir='./processed_faces_real', 
                            fake_dir='./processed_faces_fake', 
                            transform=test_transform, 
                            start_idx=7000, end_idx=9000)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=2)
model.load_state_dict(torch.load('cnn_model_state_dict_74.92.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Evaluate the model on the test set
test_correct = 0
test_total = 0

with torch.no_grad():
    for sequences, labels in tqdm(test_loader, desc='Testing', unit='batch'):
        if sequences is None:  # Skip if the image was not loaded properly
            continue

        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.2f}%')
