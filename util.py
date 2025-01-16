from __future__ import print_function

import math
import torch
import pickle
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sklearn.metrics as metrics 
from torch.utils.data import Dataset


class Plasma_Dataset(Dataset):
    def __init__(self, root_dir: str = None, file_name: str = None, transform = None) -> None:
        self.root_dir = root_dir
        self.data = self.load_pickle(file_name=file_name)
        self.transform = transform


    def load_pickle(self, file_name):
        with open(f'{self.root_dir}/{file_name}.pickle', 'rb') as pickle_file:
            content = pickle.load(pickle_file)

        return content


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        data = self.data[idx]['fft']
        label = self.data[idx]['label']

        if self.transform:
            data = self.transform(data)

        if label == 100.0:
            label = 99.0
        else:
            label = np.floor(label) 

        return (data, label)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    

class RandomPermute:
    """Create two crops of the same image"""
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        idx = torch.randperm(x.shape[self.dim])
        x_shuffled = x[:, :, idx]

        return x_shuffled
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def check_statprob(root_dir, file_name):
    dataset = Plasma_Dataset(root_dir=root_dir, file_name=file_name)
    data = dataset.__getitem__(idx=1)[0]

    list = []
    for data in dataset:
        list.append(data[0].reshape(1024*64,))
    print(np.mean(list))
    print(np.std(list))


def trend(pred, label):
    cls_pred = np.load(pred)
    cls_label = np.load(label)

    plt.scatter(x=cls_label, y=cls_pred, c='b', s=50, alpha=0.7, edgecolors= "black")
    plt.plot(cls_label, cls_label, 'r', linewidth=2.5)

    r2 = metrics.r2_score(y_pred=cls_pred, y_true=cls_label)
    rmse = metrics.root_mean_squared_error(y_pred=cls_pred, y_true=cls_label)

    print(f"R2 : {r2} | RMSE : {rmse}")

    plt.xlabel("Real RUL", fontsize = 15)
    plt.ylabel("Predicted RUL", fontsize = 15)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.show()


def lspace_vis(X=None, y=None):
    X = np.load(X)
    y = np.load(y) / 100.0

    # Create a t-SNE object to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform the data
    X_2d = tsne.fit_transform(X)

    # Plot the transformed data in 2D space
    plt.figure(figsize=(10, 7))
    # plt.plot(res[:,0], res[:,1])
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', s=50, alpha=0.7, edgecolors= "black")

    # Add a colorbar for the labels
    cbar = plt.colorbar(scatter)
    cbar.ax.tick_params(labelsize=12) 
    plt.title('2D Visualization of $y_n$ Using T-SNE', fontsize = 15)
    plt.xlabel('T-SNE Feature 1', fontsize = 15)
    plt.xticks(fontsize = 12)
    plt.ylabel('T-SNE Feature 2', fontsize = 15)
    plt.yticks(fontsize = 12)
    plt.show()
