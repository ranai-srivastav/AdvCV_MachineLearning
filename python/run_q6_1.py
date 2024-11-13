import torch
import scipy
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

if torch.cuda.is_available():
    print("CUDA!")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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


class HandwrittenDetection(nn.Module):
    def __init__(self):
        super(HandwrittenDetection, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(1024, 64),
            nn.Sigmoid(),
            nn.Linear(64, 36),
            # nn.Softmax()
        )

    def forward(self, input):
        return self.layers(input)


if __name__ == '__main__':
    train_data = NIST36_Data(type="train")
    valid_data = NIST36_Data(type="valid")
    test_data = NIST36_Data(type="test")

    num_iters = 50
    learning_rate = 1e-1
    batch_size = 32

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model = HandwrittenDetection().to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    train_acc = 0
    total_loss = 0
    plot_acc = []
    plot_loss = []

    for epoch in range(num_iters):
        model.train()
        train_acc = 0
        total_loss = 0
        for i, (tr_data, tr_label) in enumerate(trainloader):
            optimizer.zero_grad()
            tr_data = tr_data.to(device=device)
            tr_label = tr_label.to(device=device)
            model_out = model(tr_data)

            my_loss = calc_loss(tr_label, torch.argmax(model_out))
            loss = loss_func(model_out, tr_label).to(device=device)
            # loss = calc_loss(label, model_out)
            total_loss += loss.item()
            train_acc += calc_accuracy(tr_label, model_out).item()


            loss.backward()
            optimizer.step()
            # TODO is this required: model.zero_grad()

            # with torch.no_grad():
            #     for params in model.parameters():
            #         params.data.sub_(params.grad.data * learning_rate)
        train_loss = total_loss / len(trainloader)
        train_acc /= len(trainloader)
        plot_loss.append(train_loss)
        plot_acc.append(train_acc)
        print(f"TRAIN: EPOCH {epoch + 1}: TRAIN Loss = {train_loss:.4f}, TRAIN Acc = {train_acc * 100.0:.6f}")

        ############ VALIDATION ####################
        vldn_acc = 0
        total_loss = 0
        model.eval()
        for i, (vldn_data, vldn_label) in enumerate(validloader):
            vldn_data = vldn_data.to(device=device)
            vldn_label = vldn_label.to(device=device)
            model_out = model(vldn_data)

            loss = loss_func(model_out, vldn_label).to(device=device)
            total_loss += loss.item()
            vldn_acc += calc_accuracy(vldn_label, model_out)

        vldn_loss = total_loss / len(validloader)
        vldn_acc /= len(validloader)
        print(f"VALDn: EPOCH {epoch + 1}: VALDN Loss = {vldn_loss:.4f}, VALDN Acc = {vldn_acc * 100.0:.6f}")

    ############ TESTING ####################
    test_acc = 0
    test_loss = 0
    model.eval()
    for i, (test_data, test_label) in enumerate(testloader):
        test_data = test_data.to(device=device)
        test_label = test_label.to(device=device)
        model_out = model(test_data)

        loss = loss_func(model_out, test_label).to(device=device)
        total_loss += loss.item()
        test_acc += calc_accuracy(test_label, model_out)

    test_loss = total_loss / len(testloader)
    test_acc /= len(testloader)
    print("-"*100)
    print(f"TESTING: TEST Loss = {test_loss:.4f}, TEST Acc = {test_acc * 100.0:.6f}")
    ##################################

# Plotting accuracy graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), plot_acc, marker='o', color='blue')
plt.title('Model Accuracy')
plt.xlabel('Num. of epochs')
plt.ylabel('Batch accuracy')
plt.grid(True)
# plt.savefig('accuracy_graph.png')
plt.show()

# Plotting loss graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), plot_loss, marker='o', color='red')
plt.title('Model Loss')
plt.xlabel('Num. of epochs')
plt.ylabel('Avg Batch Loss')
plt.grid(True)
# plt.savefig('loss_graph.png')
plt.show()
