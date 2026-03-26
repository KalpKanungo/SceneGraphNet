import torch
from torch.utils.data import DataLoader
from src.dataset import RelationshipDataset
from src.model import RelationshipNet

device = "mps" if torch.backends.mps.is_available() else "cpu"

dataset = RelationshipDataset(
    image_dir="data/relationship_dataset/images",
    label_path="data/relationship_dataset/labels_encoded.json"
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

num_classes = len(set([item["label"] for item in dataset.data]))

model = RelationshipNet(num_classes).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/relationship_model.pth")