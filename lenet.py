import torch
from torch import nn, optim
from torch.nn import functional as F
import os

class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.args = args
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.99))
        self.apply(self.weight_init)
        self.test_acc = []
        self.best_accuracy = 0.0

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            import math
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        features = []
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        features.append(x)  # C1
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        features.append(x)  # C3
        x = F.max_pool2d(x, 2)
        features.append(x)  # S4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, features

    def train_(self, train_loader, epoch):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            output = self(data)[0]
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # print statistics
            if batch_idx % 64 == 63:
                total_loss += loss.item()
                print('[{}, {:5d}] Loss: {:.3f}'.format(
                    epoch, batch_idx + 1, total_loss / 64))
                total_loss = 0.0

    def test_(self, test_loader, epoch):
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = self(data)[0]
                # sum up batch loss
                # get the index of the max log-probability
                predict = output.data.max(1, keepdim=True)[1]
                correct += predict.eq(target.data.view_as(predict)).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        print('\nTest Accuracy: {:.2f}%\n'.format(
            accuracy))
        self.test_acc.append(accuracy)

        if accuracy > self.best_accuracy:
            self.save_best('./result')
            self.best_accuracy = accuracy

    def make_soft_labels(self, test_loader):
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = self(data)[0]
                # sum up batch loss
                # get the index of the max log-probability
                predict = output.data.max(1, keepdim=True)[1]
                correct += predict.eq(target.data.view_as(predict)).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        print('\nTest Accuracy: {:.2f}%\n'.format(
            accuracy))

    def save_best(self, path):
        try:
            os.makedirs('./result')
        except:
            print('directory ./result already exists')

        with open(os.path.join(path, 'lenet5_best.json'), 'wb') as output_file:
            torch.save(self.state_dict(), output_file)