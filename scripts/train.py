import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import simplegrad as sg
import simplegrad.module as snn

from simplegrad import Tensor, ops
from simplegrad.module import Module
from simplegrad.optim import NaiveSGD, Adam
import random

# Custom dataset for decimal to binary conversion
class DecimalToBinaryDataset(Dataset):
    def __init__(self, decimal_range=4096):
        self.decimal_range = decimal_range
        self.max_bits = int(np.log2(decimal_range)) + 1  # Number of bits needed
        self.max_decimal_digits = len(str(decimal_range - 1))  # Max number of decimal digits
        self.data = self.generate_data()

    def generate_data(self):
        data = []
        for i in range(self.decimal_range):
            # Convert to one-hot encoded tensors
            decimal_str = str(i).zfill(self.max_decimal_digits)
            binary_str = format(i, f'0{self.max_bits}b')

            # Create one-hot encoded tensors for decimal digits
            decimal_tensor = torch.zeros(self.max_decimal_digits, 10)  # 10 possible values (0-9)
            for digit_idx, digit in enumerate(decimal_str):
                decimal_tensor[digit_idx, int(digit)] = 1.0

            binary_tensor = torch.tensor(
                [int(bit) for bit in binary_str], dtype=torch.float32
            )

            data.append({
                "decimal": decimal_tensor.view(-1),
                "binary": binary_tensor.view(-1)}
            )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_max_bits(self):
        return self.max_bits

    def get_max_decimal_digits(self):
        return self.max_decimal_digits

# Data utility functions
def generate_and_split_data(decimal_range=4096, train_ratio=0.8, batch_size=64):
    dataset = DecimalToBinaryDataset(decimal_range=decimal_range)

    # Calculate sizes
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.get_max_bits(), dataset.get_max_decimal_digits()

class DecimalToBinaryNet(snn.Module):
    def __init__(
        self,
        hidden_size,
        input_digits, input_classes,
        output_bits, output_classes
    ):
        super(DecimalToBinaryNet, self).__init__()
        self.input_size = input_digits * input_classes
        self.output_size = output_bits * output_classes
        self.hidden_size = hidden_size
        # BSZ, 13, 2.
        self.output_shape = (-1, output_bits, output_classes)

        self.model = snn.Sequential(
            snn.Linear(self.input_size, self.hidden_size),
            snn.ReLU(),
            snn.Linear(self.hidden_size, self.output_size)
        )

    def __call__(self, x):
        output = self.model(x)
        output = ops.reshape(output, self.output_shape)
        return output

# Evaluation function
def evaluate(model, optimizer, test_loader):
    num_correct_numbers = 0
    num_correct_digits = 0
    total_numbers = 0
    total_digits = 0

    for batch in test_loader:
        decimal_data = batch["decimal"]
        decimal_data = Tensor(
            decimal_data.detach().cpu().numpy(),
            requires_grad=False
        )
        binary_data = batch["binary"]
        batch_target = Tensor(
            binary_data.detach().cpu().numpy(),
            requires_grad=False
        )

        predictions = model(decimal_data)

        _, predicted_indices = torch.tensor(predictions.data).max(dim=2)
        target_indices = torch.tensor(batch_target.data)

        # Count fully correct predictions (all bits must be correct)
        num_correct_numbers += (predicted_indices == target_indices).all(dim=1).sum().item()
        total_numbers += decimal_data.shape[0]

        # Count correct individual digits
        num_correct_digits += (predicted_indices == target_indices).sum().item()
        total_digits += decimal_data.shape[0] * predicted_indices.shape[1]

    # Calculate accuracies
    number_accuracy = 100 * num_correct_numbers / total_numbers
    digit_accuracy = 100 * num_correct_digits / total_digits

    optimizer.zero_grad()
    return number_accuracy, digit_accuracy


def train(model, criterion, optimizer, num_epochs, log_steps=50):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            decimal_data = batch["decimal"]
            decimal_data = Tensor(
                decimal_data.detach().cpu().numpy(),
                requires_grad=False
            )
            binary_data = batch["binary"]
            batch_target = Tensor(
                binary_data.detach().cpu().numpy(),
                requires_grad=False
            )

            optimizer.zero_grad()
            batch_logits = model(decimal_data)

            flattened_logits = ops.reshape(batch_logits, (-1, output_classes))
            flattened_target = ops.reshape(batch_target, -1)

            loss = criterion(flattened_logits, flattened_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.data

        if epoch % log_steps == 0:
            number_accuracy, digit_accuracy = evaluate(model, optimizer, test_loader)
            epoch_loss = running_loss / len(train_loader)
            print(
                f"Epoch {epoch}: Training loss: {epoch_loss:.4f}, "
                f"Number accuracy: {number_accuracy:.2f}%, "
                f"Digit accuracy: {digit_accuracy:.2f}"
            )

def illustrate(model, batch):
    indices = [i for i in range(batch['decimal'].shape[0])]
    random.shuffle(indices)

    for i, idx in enumerate(indices[:10]):
        decimal_example = batch["decimal"][idx]
        decimal = decimal_example.reshape(-1, 10).max(dim=1).indices
        decimal = [int(i.item()) for i in decimal]

        logits = model(decimal_example)
        predicted = torch.tensor(logits.data).reshape(-1, 2).max(dim=1).indices
        predicted = [int(i.item()) for i in predicted]

        binary_example = batch["binary"][idx]
        target = [int(i.item()) for i in binary_example]

        print(
            f"Example {i+1}:\n"
            f"  Decimal number  : {decimal}\n"
            f"  Target binary   : {target}\n"
            f"  Predicted binary: {predicted}\n"
        )

if __name__ == "__main__":

    decimal_range = 2048
    train_loader, test_loader, max_bits, max_decimal_digits = generate_and_split_data(
        decimal_range=decimal_range,
        train_ratio=0.9,
        batch_size=64
    )

    # Initialize the model
    input_digits = max_decimal_digits
    input_classes = 10  # 0-9 for decimal digits
    hidden_size = 128
    output_bits = max_bits
    output_classes = 2  # 0-1 for binary digits

    model = DecimalToBinaryNet(
        hidden_size=hidden_size,
        input_digits=input_digits,
        input_classes=input_classes,
        output_bits=output_bits,
        output_classes=output_classes
    )

    criterion = snn.CrossEntropyLoss()
    # optimizer = NaiveSGD(model.parameters(), lr=0.04)
    optimizer = Adam(model.parameters(), lr=0.001)
    num_epochs = 4000

    train(model, criterion, optimizer, num_epochs, log_steps=50)

    print("\nIllustration:")
    batch = next(iter(test_loader))
    illustrate(model, batch)
