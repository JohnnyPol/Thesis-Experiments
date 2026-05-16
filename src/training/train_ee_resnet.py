from contextlib import nullcontext
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.loaders import data_loader
from src.models.blocks import ResidualBlock
from src.models.resnet_ee import ResNetEE18


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_context():
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def _validate_exit_weights(exit_weights):
    if len(exit_weights) != 4:
        raise ValueError("exit_weights must contain 4 values: [exit0, exit1, exit2, final]")
    weight_sum = sum(exit_weights)
    if weight_sum == 0:
        raise ValueError("exit_weights must not sum to zero")
    return weight_sum


def _forward_all_exits(model, images):
    x = model.conv1(images)
    x = model.maxpool(x)

    x0 = model.layer0(x)
    out0 = model.exit0(x0)

    x1 = model.layer1(x0)
    out1 = model.exit1(x1)

    x2 = model.layer2(x1)
    out2 = model.exit2(x2)

    x3 = model.layer3(x2)
    xf = model.avgpool(x3)
    xf = torch.flatten(xf, 1)
    out_final = model.fc(xf)

    return [out0, out1, out2, out_final]


def evaluate_ee_validation(model, valid_loader, criterion, exit_weights=None):
    model.eval()

    if exit_weights is None:
        exit_weights = [1.0, 1.0, 1.0, 1.0]
    weight_sum = _validate_exit_weights(exit_weights)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    exit_correct = [0, 0, 0, 0]

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with _autocast_context():
                outputs = _forward_all_exits(model, images)
                losses = [criterion(o, labels) for o in outputs]

                weighted_loss = sum(w * l for w, l in zip(exit_weights, losses)) / weight_sum
                total_loss += weighted_loss.item() * labels.size(0)

            final_preds = outputs[-1].argmax(dim=1)
            total_correct += (final_preds == labels).sum().item()
            total_samples += labels.size(0)

            for i, out in enumerate(outputs):
                preds_i = out.argmax(dim=1)
                exit_correct[i] += (preds_i == labels).sum().item()

    avg_loss = total_loss / total_samples
    final_acc = 100.0 * total_correct / total_samples
    exit_accs = [100.0 * c / total_samples for c in exit_correct]

    return avg_loss, final_acc, exit_accs


def train_ee_first_term_only(
    model,
    epochs,
    train_loader,
    valid_loader=None,
    lr=1e-3,
    exit_weights=None
):
    """
    Train an early-exit model using only the first term:
    CrossEntropyLoss on each exit.

    Args:
        model: ResNetEE18 model
        epochs: number of epochs
        train_loader: training dataloader
        valid_loader: optional validation dataloader
        lr: learning rate
        exit_weights: optional list of 4 weights for [exit0, exit1, exit2, final]

    Returns:
        Trained model
    """
    if exit_weights is None:
        exit_weights = [1.0, 2.0, 3.0, 4.0]
    weight_sum = _validate_exit_weights(exit_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_acc = 0.0
    best_state_dict = None

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_final_correct = 0
        running_exit_correct = [0, 0, 0, 0]

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with _autocast_context():
                outputs = model(images)   # [out0, out1, out2, out_final]
                losses = [criterion(o, labels) for o in outputs]

                total_loss = sum(w * l for w, l in zip(exit_weights, losses)) / weight_sum

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            running_loss += total_loss.item() * batch_size
            running_total += batch_size

            for i, out in enumerate(outputs):
                preds_i = out.argmax(dim=1)
                running_exit_correct[i] += (preds_i == labels).sum().item()

            final_preds = outputs[-1].argmax(dim=1)
            running_final_correct += (final_preds == labels).sum().item()

            loop.set_postfix(
                loss=running_loss / running_total,
                final_acc=100.0 * running_final_correct / running_total,
                exit0_acc=100.0 * running_exit_correct[0] / running_total,
                exit1_acc=100.0 * running_exit_correct[1] / running_total,
                exit2_acc=100.0 * running_exit_correct[2] / running_total
            )

        train_loss = running_loss / running_total
        train_final_acc = 100.0 * running_final_correct / running_total
        train_exit_accs = [100.0 * c / running_total for c in running_exit_correct]

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Train Exit0 Acc: {train_exit_accs[0]:.2f}%")
        print(f"Train Exit1 Acc: {train_exit_accs[1]:.2f}%")
        print(f"Train Exit2 Acc: {train_exit_accs[2]:.2f}%")
        print(f"Train Final Acc: {train_final_acc:.2f}%")

        if valid_loader is not None:
            val_loss, val_final_acc, val_exit_accs = evaluate_ee_validation(
                model=model,
                valid_loader=valid_loader,
                criterion=criterion,
                exit_weights=exit_weights
            )

            print(f"Val Loss       : {val_loss:.4f}")
            print(f"Val Exit0 Acc  : {val_exit_accs[0]:.2f}%")
            print(f"Val Exit1 Acc  : {val_exit_accs[1]:.2f}%")
            print(f"Val Exit2 Acc  : {val_exit_accs[2]:.2f}%")
            print(f"Val Final Acc  : {val_final_acc:.2f}%")

            if val_final_acc > best_val_acc:
                best_val_acc = val_final_acc
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    end_time = time.time()
    print(f"\nTraining time: {end_time - start_time:.2f} sec")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


if __name__ == "__main__":
    train_loader, valid_loader = data_loader(data_dir="./data", batch_size=64)
    test_loader = data_loader(data_dir="./data", batch_size=1, test=True)

    model_ee18 = ResNetEE18(ResidualBlock, [2, 2, 2, 2], num_classes=10).to(device)

    model_ee18 = train_ee_first_term_only(
        model=model_ee18,
        epochs=40,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=1e-3,
        exit_weights=[1.0, 2.0, 3.0, 4.0],
    )

    torch.save(model_ee18.state_dict(), "resnet18_ee_first_term.pth")
