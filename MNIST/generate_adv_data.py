import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os.path as osp
from models import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate adv data')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path for adv model')
    parser.add_argument('--eps', type=float, default=0.3, help='eps for data')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    args.device = torch.device("cuda" if use_cuda else "cpu")

    atkmodel = Autoencoder().to(args.device)
    atkmodel.load_state_dict(torch.load(args.model_path))
    atkmodel.eval()
    print('load from ' + args.model_path)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=10000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=10000, shuffle=True)

    traindata, testdata = None, None
    for data, _ in train_loader:
        traindata = data
        break
    for data, _ in test_loader:
        testdata = data
        break

    traindata_adv = torch.zeros_like(traindata).float()
    testdata_adv = torch.zeros_like(testdata).float()

    for idx, i in enumerate(traindata):
        data_float = i.float()[None, :, :].to(args.device)/255.
        with torch.no_grad():
            noise = atkmodel(data_float) * args.eps
            atkdata = torch.clamp(data_float + noise, 0, 1)
        traindata_adv[idx] = atkdata[0].cpu()

    for idx, i in enumerate(testdata):
        data_float = i.float()[None, :, :].to(args.device)/255.
        with torch.no_grad():
            noise = atkmodel(data_float) * args.eps
            atkdata = torch.clamp(data_float + noise, 0, 1)
        testdata_adv[idx] = atkdata[0].cpu()

    torch.save(traindata_adv, 'training_adv.pt')
    torch.save(testdata_adv, 'test_adv.pt')
