import torch
import scipy
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


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
            data = data.reshape([data.shape[0], 3, 32, 32])
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
    plt.plot(range(1, num_epochs + 1), plot_acc, marker='o', color='blue')
    plt.title('Model Accuracy')
    plt.xlabel('Num. of epochs')
    plt.ylabel('Batch accuracy')
    plt.grid(True)
    # plt.savefig('accuracy_graph.png')
    plt.show()

    # Plotting loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), plot_loss, marker='o', color='red')
    plt.title('Model Loss')
    plt.xlabel('Num. of epochs')
    plt.ylabel('Avg Batch Loss')
    plt.grid(True)
    # plt.savefig('loss_graph.png')
    plt.show()


def val_test(model, data_loader, device, num_epochs):
    fin_acc = 0
    total_loss = 0
    model.eval()
    for batch_num, (data, label) in enumerate(data_loader):

        data = data.to(device=device)
        data = data.reshape([data.shape[0], 3, 32, 32])
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
            nn.Conv2d(3, 16, 5, padding=5//2),
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
    num_iters = 10
    learning_rate = 5e-2
    batch_size = 16

    if torch.cuda.is_available():
        print("CUDA!")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Curr device = {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model = HandwrittenDetectionCNN().to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    model.train()
    train(model, trainloader, device, num_iters)

    model.eval()
    # val_test(model, validloader, device, 1)
    val_test(model, testloader, device, 1)

