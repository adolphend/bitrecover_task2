from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py
import torch.nn as nn
import numpy as np
import torch
import BER_calc
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import copy
import argparse


np.random.seed(100)
torch.manual_seed(100)
debug = True


def parse_args():
    parser = argparse.ArgumentParser(description='Bit corrections')
    parser.add_argument('--train_data', type=str, default='train.mat', help='path to train dataset')
    parser.add_argument('--test_data', type=str, default='test.mat', help='path to test dataset')
    parser.add_argument('--batchsize', type=int, default=4048, help='batchsize')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('--cuda', type=bool, default=True, help='use gpu')
    parser.add_argument('--input_size', type=int, default=4, help='input size')
    parser.add_argument('--N', type=int, default=20, help='neighbouring symbols')
    parser.add_argument('--hidden_size', type=int, default=226, help='hidden size of lstm')
    parser.add_argument('--bidirectional', type=bool, default=True, help='lstm bidirectional')
    parser.add_argument('--output_size', type=int, default=2, help='number of output size')
    parser.add_argument('--output_file', type=str, default='output.npy', help='logs file')
    parser.add_argument('--output_model', type=str, default='model.pth', help='weight file')
    parser.add_argument('--epochs', type=int, default=200, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    return parser.parse_args()


def BER_est(x_in, x_ref):
    QAM_order = 16
    return BER_calc.QAM_BER_gray(x_in, x_ref, QAM_order)


def loadmat(filename):
    data = {}
    with h5py.File('Dataset1.mat', 'r') as f:
        for k, v in f.items():
            data[k] =  np.asarray(v)
    return data


class ErrorCorrectionModel(nn.Module):
    def __init__(self, output_size=2,
                 input_size=4, hidden_size=226,
                 bidirectional=True, dropout=0,
                 batch_first=True, sequence_length=41):
        super(ErrorCorrectionModel, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=batch_first)
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        self.linear1 = nn.Linear(sequence_length * num_directions * hidden_size, 100)
        self.linear2 = nn.Linear(100, output_size)
        self.fc = nn.ReLU()

    def forward(self, x):
        x, _ = self.rnn(x)
        batch, sq, _ = x.shape
        x = x.reshape(batch, -1)
        x = self.fc(self.linear1(x))
        x = self.linear2(x)
        return x



class MatDataset(Dataset):
    def __init__(self, mat, N):
        self.X = [list(x[0])  + list(y[0]) for x, y in zip(mat["polX_in"], mat["polY_in"])]
        self.X = np.pad(np.asarray(self.X), ((N, N), (0, 0)))
        self.N = N
        self.Y = mat["polX_desired"]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X = self.X[idx:idx + 2 * self.N + 1, :]
        y = self.Y[idx][0]
        return X, np.asarray(list(y))


def train_model(model, criterion, optimizer, dataloaders, num_epochs=25,
                device="cuda", writer=None, model_path=None):
    BER = {k: [] for k, v in dataloaders.items()}
    MSE = {k: [] for k, v in dataloaders.items()}
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes = {k: len(v.dataset)for k, v in dataloaders.items()}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            BER_list = []
            # Iterate over data.
            n = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #print(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
                for x_in, x_ref in zip(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()):
                    BER_list.append(BER_est(x_in, x_ref))
                    running_corrects = running_corrects + BER_est(x_in, x_ref)
                writer.add_scalar("Loss/" + phase, loss, epoch)
                n += len(labels)
                writer.add_scalar("Accuracy/" + phase,
                                  running_corrects / n, epoch)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(np.mean(np.asarray(BER_list)))
            print('{} Loss: {:.4f} Acc: {:e}'.format(
                phase, epoch_loss, epoch_acc))
            BER[phase].append(epoch_acc)
            MSE[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print()
    torch.save(best_model_wts, model_path)
    return BER, MSE


def main():
    args = parse_args()
    datatrain = loadmat(args.train_data)
    model = ErrorCorrectionModel(output_size=args.output_size,
                                 input_size=args.input_size,
                                 hidden_size=args.hidden_size,
                                 bidirectional=args.bidirectional,
                                 sequence_length= 2 * args.N + 1)
    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)
    N = args.N
    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    trainset = MatDataset(datatrain, N)
    trainloader = DataLoader(trainset,
                             batch_size=args.batchsize,
                             shuffle=True,
                             num_workers=args.num_workers)
    datatest = loadmat(args.test_data)
    testset = MatDataset(datatest, N)
    testloader = DataLoader(trainset,
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=args.num_workers)
    writer = SummaryWriter()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataloaders = {}
    dataloaders["val"] = testloader
    dataloaders["train"] = trainloader
    something = train_model(model,
                            criterion,
                            optimizer,
                            dataloaders,
                            num_epochs=args.epochs,
                            device=device,
                            writer=writer,
                            model_path=args.output_model)
    with open(args.output_file, 'wb') as f:
        np.save(f, something)


"""
def main():
    datatrain = loadmat('Dataset1.mat')
    model = ErrorCorrectionModel()
    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)
    N = 20
    device = "cuda"
    model.to(device)
    trainset = MatDataset(datatrain, N)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=16)
    datatest = loadmat('Dataset3.mat')
    testset = MatDataset(datatest, N)
    testloader = DataLoader(trainset, batch_size=16, shuffle=False, num_workers=16)
    writer = SummaryWriter()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    dataloaders = {}
    dataloaders["val"] = testloader
    dataloaders["train"] = trainloader
    something = train_model(model, criterion, optimizer, dataloaders, num_epochs=200, device=device, writer=writer)
    with open("something_test.npy", 'wb') as f:
        np.save(f, something)
    datatrain = loadmat('Dataset2.mat')
    model = ErrorCorrectionModel()
    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal(param)
    N = 20
    device = "cuda"
    model.to(device)
    trainset = MatDataset(datatrain, N)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=16)
    datatest = loadmat('Dataset3.mat')
    testset = MatDataset(datatest, N)
    testloader = DataLoader(trainset, batch_size=16, shuffle=False, num_workers=16)
    writer = SummaryWriter()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    dataloaders = {}
    dataloaders["val"] = testloader
    dataloaders["train"] = trainloader
    someth = train_model(model, criterion, optimizer, dataloaders, num_epochs=200, device=device, writer=writer)
    with open("someth_test.npy", 'wb') as f:
        np.save(f, someth)
"""


if __name__ == "__main__":
    main()
