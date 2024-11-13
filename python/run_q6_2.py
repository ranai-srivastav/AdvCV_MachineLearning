import torch
import scipy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


class NIST36_Data(torch.utils.data.Dataset):
    def __init__(self, type):
        self.type = type
        self.data = scipy.io.loadmat(f"../data/nist36_{type}.mat")
        self.inputs, self.one_hot_target = (
            self.data[f"{self.type}_data"],
            self.data[f"{self.type}_labels"],
        )
        self.target = np.argmax(self.one_hot_target, axis=1)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.inputs[index]).type(torch.float32)
        target = torch.tensor(self.target[index]).type(torch.LongTensor)
        return inputs, target


def calc_accuracy(y, probs):
    predictions = torch.argmax(probs, dim=1)
    acc = (predictions == y).float().mean()
    return acc


def calc_loss(y, probs):
    loss = -torch.sum(torch.multiply(y, torch.log(probs)))
    return loss


def train(model, data_loader, device, num_epochs):
    model.train()
    plot_acc = []
    plot_loss = []
    for epoch in range(num_epochs):
        fin_acc = 0
        total_loss = 0
        for batch_num, (data, label) in enumerate(data_loader):

            optimizer.zero_grad()
            data = data.to(device=device)
            data = data.reshape([data.shape[0], 1, 32, 32])
            label = label.to(device=device)
            model_out = model(data)

            loss = loss_func(model_out, label).to(device=device)
            total_loss += loss.item()
            fin_acc += calc_accuracy(label, model_out).item()

            loss.backward()
            optimizer.step()

        fin_loss = total_loss / len(data_loader)
        fin_acc /= len(data_loader)
        plot_loss.append(fin_loss)
        plot_acc.append(fin_acc)
        print(f"TRAIN: EPOCH {epoch + 1}: TRAIN Loss = {fin_loss:.4f}, TRAIN Acc = {fin_acc * 100.0:.6f}")

    # Plotting accuracy graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), plot_acc, marker='o', color='blue')
    plt.title('Model Accuracy')
    plt.xlabel('Num. of epochs')
    plt.ylabel('Batch accuracy')
    plt.grid(True)
    # plt.savefig('accuracy_graph.png')
    plt.show()

    # Plotting loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), plot_loss, marker='o', color='red')
    plt.title('Model Loss')
    plt.xlabel('Num. of epochs')
    plt.ylabel('Avg Batch Loss')
    plt.grid(True)
    # plt.savefig('loss_graph.png')
    plt.show()
    plt.show()


def val_test(model, data_loader, device, num_epochs):
    fin_acc = 0
    total_loss = 0
    model.eval()
    for batch_num, (data, label) in enumerate(data_loader):

        data = data.to(device=device)
        data = data.reshape([data.shape[0], 1, 32, 32])
        label = label.to(device=device)
        model_out = model(data)

        loss = loss_func(model_out, label).to(device=device)
        total_loss += loss.item()
        fin_acc += calc_accuracy(label, model_out)


    fin_loss = total_loss / len(data_loader)
    fin_acc /= len(data_loader)
    print(f"METRIC: Loss = {fin_loss:.4f}, Acc = {fin_acc * 100.0:.6f}")

class HandwrittenDetectionCNN(nn.Module):
    def __init__(self):
        super(HandwrittenDetectionCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=5//2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(16, 32, 5, padding=5//2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 36)
        )

    def forward(self, input):
        return self.layers(input)


if __name__ == '__main__':
    train_data = NIST36_Data(type="train")
    valid_data = NIST36_Data(type="valid")
    test_data = NIST36_Data(type="test")

    num_iters = 40
    learning_rate = 5e-2
    batch_size = 16

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model = HandwrittenDetectionCNN().to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    model.train()
    train(model, trainloader, device, 15)

    model.eval()
    val_test(model, validloader, device, 1)
    val_test(model, testloader, device, 1)

