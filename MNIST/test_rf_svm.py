from sklearn.ensemble import RandomForestClassifier
import torch
from torchvision import datasets, transforms
from sklearn import svm

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

Xtrain_clean, ytrain, Xtest_clean, ytest = None, None, None, None
for data, labels in train_loader:
    Xtrain_clean = data
    ytrain = labels
    break
for data, labels in test_loader:
    Xtest_clean = data
    ytest = labels
    break

Xtrain_adv = torch.load('training_adv.pt')
Xtrain_clean = Xtrain_clean.float().numpy() / 255.
ytrain = ytrain.numpy()
Xtrain_adv = Xtrain_adv.numpy()

Xtest_adv = torch.load('test_adv.pt')
Xtest_clean = Xtest_clean.float().numpy() / 255.
ytest = ytest.numpy()
Xtest_adv = Xtest_adv.numpy()

print('=============SVM===============')
svc = svm.SVC(C=200, kernel='rbf', gamma=0.01, cache_size=8000,
              probability=False)
svc.fit(Xtrain_adv.reshape(60000, -1), ytrain)
print('train on adv, test on clean: ',
      svc.score(Xtest_clean.reshape(10000, -1), ytest))
svc = svm.SVC(C=200, kernel='rbf', gamma=0.01, cache_size=8000,
              probability=False)
svc.fit(Xtrain_clean.reshape(60000, -1), ytrain)
print('train on clean, test on clean: ',
      svc.score(Xtest_clean.reshape(10000, -1), ytest))

print('=============RF===============')
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)
rfc.fit(Xtrain_adv.reshape(60000, -1), ytrain)
print('train on adv, test on clean: ',
      rfc.score(Xtest_clean.reshape(10000, -1), ytest))
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)
rfc.fit(Xtrain_clean.reshape(60000, -1), ytrain)
print('train on clean, test on clean: ',
      rfc.score(Xtest_clean.reshape(10000, -1), ytest))
