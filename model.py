import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.sigmoid(self.fc(out[:, -1, :]))
        return out

model = BiLSTM(input_size=FRAME_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for frames, labels in train_loader:
        outputs = model(frames)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch+1}/{10}, Batch: {i}/{len(dataloader)}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "model.pt")

# Testing the model on new video
def predict_deepfake(video_frames):
    model.eval()
    with torch.no_grad():
        # Get model prediction
        outputs = model(video_frames)
        _, predicted = torch.max(outputs.data, 1)
        prediction = predicted.cpu().numpy()[0]
        return prediction