from src.data.loaders import data_loader
from src.models.blocks import ResidualBlock
from src.models.resnet_baseline import ResNet
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, valid_loader = data_loader(data_dir="./data", batch_size=64)

test_loader = data_loader(data_dir="./data", batch_size=1, test=True)


def evaluate_baseline_validation(model, valid_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples
    return avg_loss, acc


def train_baseline(model, epochs, train_loader, valid_loader=None, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    best_state_dict = None

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        loop = tqdm(
            train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"
        )

        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            loop.set_postfix(
                loss=running_loss / running_total,
                acc=100.0 * running_correct / running_total,
            )

        train_loss = running_loss / running_total
        train_acc = 100.0 * running_correct / running_total

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        if valid_loader is not None:
            val_loss, val_acc = evaluate_baseline_validation(
                model, valid_loader, criterion
            )
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} sec")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


model_baseline_18 = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10).to(device)

model_baseline_18 = train_baseline(
    model=model_baseline_18,
    epochs=40,
    train_loader=train_loader,
    valid_loader=valid_loader,
    lr=1e-3,
)

torch.save(model_baseline_18.state_dict(), "resnet18_baseline.pth")
