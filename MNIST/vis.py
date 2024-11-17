import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=10000, shuffle=True)

traindata = None
for data, _ in train_loader:
    traindata = data
    break

advdata = torch.load('training_adv.pt')
for i in range(1, 11):
    idx = np.random.randint(10000)
    fig.add_subplot(rows, columns, 2 * i - 1)
    plt.imshow(traindata[idx].squeeze())
    fig.add_subplot(rows, columns, 2 * i)
    plt.imshow(advdata[idx].squeeze())
plt.savefig('mnist.png')
