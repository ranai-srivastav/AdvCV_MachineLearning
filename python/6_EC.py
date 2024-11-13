import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import torchvision.models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import SqueezeNet, SqueezeNet1_1_Weights, squeezenet1_1


def calc_accuracy(y, probs):
    predictions = torch.argmax(probs, dim=1)
    acc = (predictions == y).float().mean()
    return acc


def calc_loss(y, probs):
    loss = -torch.sum(torch.multiply(y, torch.log(probs)))
    return loss


def train(model, data_loader, device, learning_rate, num_epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        fin_acc = 0
        total_loss = 0
        for batch_num, (data, label) in enumerate(data_loader):
            optimizer.zero_grad()
            data = data.to(device=device)
            data = data.reshape([data.shape[0], 3, 256, 256])
            label = label.to(device=device)
            model_out = model(data)

            loss = loss_func(model_out, label).to(device=device)
            total_loss += loss.item()
            fin_acc += calc_accuracy(label, model_out)

            loss.backward()
            optimizer.step()

        fin_loss = total_loss / len(data_loader)
        fin_acc /= len(data_loader)
        print(f"TRAIN: EPOCH {epoch + 1}: TRAIN Loss = {fin_loss:.4f}, TRAIN Acc = {fin_acc * 100.0:.6f}")


def val_test(model, data_loader, device, num_epochs):
    model.eval()
    fin_acc = 0
    total_loss = 0
    loss_func = nn.CrossEntropyLoss()
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


class FlowerDetCNN(nn.Module):
    def __init__(self):
        super(FlowerDetCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=5 // 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(16, 32, 5, padding=5 // 2),
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

if __name__ == "__main__":
    #####PARAMS#####
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-3
    ################

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"!!! Using Device {device}")


    #### DATA LOADING ####
    tensorTransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder("../data/oxford-flowers17/train", transform=tensorTransform)
    valid_dataset = torchvision.datasets.ImageFolder("../data/oxford-flowers17/val", transform=tensorTransform)
    test_dataset = torchvision.datasets.ImageFolder("../data/oxford-flowers17/test", transform=tensorTransform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              )

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True)


    ############# Squeeze Net ###############

    sq_net = torchvision.models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    conv = nn.Conv2d(512, 17, kernel_size=1)
    classifier = nn.Sequential(
            nn.Dropout(p=0.5), conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )
    sq_net.classifier = classifier
    sq_net.to(device=device)
    sq_net.train()

    plot_loss = []
    plot_acc = []

    optimizer = optim.SGD(sq_net.parameters(), lr=learning_rate)
    loss_fcn = nn.CrossEntropyLoss().to(device=device)

    for epoch in range(num_epochs):
        acc = 0
        tr_loss = 0
        for batch_num, (data, true_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device=device)
            true_labels = true_labels.to(device=device)

            probs = sq_net(data)

            loss = loss_fcn(probs, true_labels)
            acc += calc_accuracy(true_labels, probs).item()

            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
        print(f" TRAIN {epoch} Acc = {acc / len(train_loader)}")
        print(f" TRAIN {epoch} Loss = {tr_loss / len(train_loader)}")
        plot_acc.append(acc / len(train_loader))
        plot_loss.append(tr_loss / len(train_loader))

    # Plotting accuracy graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), plot_acc, marker='o', color='blue')
    # plt.plot(range(1, 2), plot_acc, marker='o', color='blue')
    plt.title('Model Accuracy')
    plt.xlabel('Num. of epochs')
    plt.ylabel('Batch accuracy')
    plt.grid(True)
    # plt.savefig('accuracy_graph.png')
    plt.show()

    # Plotting loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), plot_loss, marker='o', color='red')
    # plt.plot(range(1, 2), plot_loss, marker='o', color='red')
    plt.title('Model Loss')
    plt.xlabel('Num. of epochs')
    plt.ylabel('Avg Batch Loss')
    plt.grid(True)
    # plt.savefig('loss_graph.png')
    plt.show()

    ## Validating
    sq_net.eval()
    for batch_num, (data, true_labels) in enumerate(test_loader):
        data = data.to(device=device)
        true_labels = true_labels.to(device=device)

        probs = sq_net(data)

        loss = loss_fcn(probs, true_labels)
        acc += calc_accuracy(true_labels, probs).item()

        tr_loss += loss.item()
    print(f" VAL Acc = {acc / len(valid_loader)}")
    print(f" VAL Loss = {loss / len(valid_loader)}")
    #########################################

    ########## Custom Model #################
    model = FlowerDetCNN().to(device=device)
    train(model, train_loader, device, learning_rate, num_epochs)
