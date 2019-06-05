from __future__ import print_function
import os
import argparse
import pickle
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from distilled_model import DistilledSoftDecisionTree
from lenet import LeNet
from model import SoftDecisionTree

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--input-dim', type=int, default=28*28, metavar='N',
                    help='input dimension size(default: 28 * 28)')
parser.add_argument('--output-dim', type=int, default=10, metavar='N',
                    help='output dimension size(default: 10)')
parser.add_argument('--max-depth', type=int, default=8, metavar='N',
                    help='maximum depth of tree(default: 8)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 40)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lmbda', type=float, default=0.1, metavar='LR',
                    help='temperature rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--import-mode', type=bool, default=False, metavar='N',
                    help='to import lenet and mnist soft targets')
parser.add_argument('--temperature', type=int, default=20, metavar='N',
                    help='Temperature for softmax in LeNet, to be implemented')
parser.add_argument('--beta', type=float, default=0.5, metavar='N',
                    help='Inverse temperature beta')
parser.add_argument('--save-model', type=bool, default=False, metavar='N',
                    help='Save the best model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


try:
    os.makedirs('./data')
except:
    print('directory ./data already exists')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

importMode = args.import_mode
print(importMode, args.lmbda)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=False, **kwargs)

def save_result(acc):
    try:
        os.makedirs('./result')
    except:
        print('directory ./result already exists')
    filename = os.path.join('./result/', 'bp.pickle')
    f = open(filename,'wb')
    pickle.dump(acc, f)
    f.close()


class SoftMNIST(datasets.MNIST):


    def __init__(self, root, import_targets=False, init_target_transform=None, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.init_target_transform = init_target_transform
        if import_targets:
            filename = os.path.join(root, 'soft_targets.pickle')
            f = open(filename, 'rb')
            self.train_labels = pickle.load(f)
            f.close()
        else:
            self.train_labels = [self.get_soft_label(x) for x in self.train_data]
            filename = os.path.join(root, 'soft_targets.pickle')
            f = open(filename, 'wb')
            pickle.dump(self.train_labels, f)
            f.close()

    def get_soft_label(self, img):
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return self.init_target_transform(img)


leNetModel = LeNet(args)

if args.cuda:
    leNetModel.cuda()

soft_labels = []

if importMode:
    leNetModel.load_state_dict(torch.load(os.path.join('./result', 'lenet5_best.json')))
else:
    for epoch in range(1, args.epochs + 1):
        leNetModel.train_(train_loader, epoch)


def get_soft_label(img):
    return leNetModel.forward_with_temperature(img.view(1, *(img.size()))).view(-1)


soft_train_loader = torch.utils.data.DataLoader(
    SoftMNIST('./data', import_targets=True, train=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]), init_target_transform=get_soft_label),
    batch_size=args.batch_size, **kwargs)

def show(data_loader):
    images, foo = next(iter(data_loader))
    print(foo)
    from torchvision.utils import make_grid
    npimg = make_grid(images, normalize=True, pad_value=.5).numpy()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=((13, 5)))
    import numpy as np
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.setp(ax, xticks=[], yticks=[])

    return fig, ax

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = SoftDecisionTree(args)

if args.cuda:
    model.cuda()

# for epoch in range(1, args.epochs + 1):
#     model.train_(train_loader, epoch)


dist_model = DistilledSoftDecisionTree(args)

if args.cuda:
    dist_model.cuda()

for epoch in range(1, args.epochs + 1):
    dist_model.train_(soft_train_loader, epoch)

# model.test_(test_loader, 1)
dist_model.test_(test_loader, 1)
leNetModel.test_(test_loader, 1)
print("\n --------- Best accuracy --------")
print("LeNet: {:.4f}%".format(leNetModel.best_accuracy))
print("SoftDecisionTree: {:.4f}%".format(model.best_accuracy))
print("DistilledSoft: {:.4f}%".format(dist_model.best_accuracy))

fig, ax = show(soft_train_loader)
plt.show()
exit()
# save_result(model)
