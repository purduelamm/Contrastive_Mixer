from __future__ import print_function


import torch
import numpy as np
from torchvision import transforms
import torch.backends.cudnn as cudnn

from util import Plasma_Dataset
from networks.mixer import MLPMixer


def set_loader(opt):
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((opt.mean),(opt.std))]) # -86.617739, 16.398298
    train_dataset = Plasma_Dataset(root_dir='./data/val', file_name='combined', transform=train_transform)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=(train_sampler is None),
        num_workers=16, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(path):
    checkpoint = torch.load(path)
    model = MLPMixer(image_size=(1024, 64), channels=1, patch_size=16, dim=256, depth=4, num_classes=256)
    model.load_state_dict(checkpoint['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    return model


def testing(train_loader, model):
    model.eval()

    feat_list = []
    label_list = []

    for _, (images, labels) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        outputs = model(images.float())
        outputs = outputs.detach().cpu().numpy()
        feat_list.append(outputs)
        labels = labels.detach().cpu().numpy()
        label_list.append(labels)
    
    feat_list = np.concatenate(feat_list, axis=0)
    label_list = np.array(label_list)

    return feat_list, label_list


def main():
    train_loader = set_loader()

    model = set_model()
    save_res = testing(train_loader, model)
    np.save('./outputs/save_feats.npy', save_res[0])
    np.save('./outputs/save_labels.npy', save_res[1])


if __name__ == '__main__':
    main()
