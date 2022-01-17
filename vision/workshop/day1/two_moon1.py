import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from sklearn.datasets import make_moons
from matplotlib import pyplot
import matplotlib.pyplot as plt

import torch.utils.data as data

BATCH_SIZE = 100
TEST_BATCH_SIZE = 3000

class TwoMoon(data.Dataset):
    def __init__(self, train=True):
        super(TwoMoon,self).__init__()
        self.train = train
        if(self.train==True):
            self.train_data, self.train_labels = make_moons(n_samples=3000, noise=0.15)
            # self.train_data = self.train_data.float()
        else:
            self.test_data, self.test_labels = make_moons(n_samples=TEST_BATCH_SIZE, noise=0.15)

    def __getitem__(self,index):
        if self.train:
            input_data= self.train_data[index]
            target = self.train_labels[index]
        else:
            input_data= self.test_data[index]
            target = self.test_labels[index]

        return input_data, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)




trainset = TwoMoon(train=True)
testset = TwoMoon(train=False)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
                                         shuffle=False, num_workers=2)


pos_mask = trainset.train_labels == 1
neg_mask = trainset.train_labels == 0

plt.figure()
plt.plot(trainset.train_data[pos_mask, 0], trainset.train_data[pos_mask, 1], '.', color='b', markersize=1)
plt.plot(trainset.train_data[neg_mask, 0], trainset.train_data[neg_mask, 1], '.', color='r', markersize=1)
plt.show(block=False)
plt.close()



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.d1 = nn.Linear(2, 30)
        self.d2 = nn.Linear(30, 30)
        self.d3 = nn.Linear(30, 1)

    def forward(self, x):

        x = x.flatten(start_dim = 1)

        x = self.d1(x)
        x = F.relu(x)

        x = self.d2(x)
        x = F.relu(x)

        logits = self.d3(x)
        out = torch.sigmoid(logits)
        return out

#SGD : 0.1 / Adam : 0.001
learning_rate = 0.001
num_epochs = 10


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel()
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    pos_corrects = ((logit>0.5) & (target.data==1)).float()
    neg_corrects = ((logit<=0.5) & (target.data==0)).float()
    corrects = pos_corrects.sum() + neg_corrects.sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

    
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    model = model.train()

    ## training step
    for i, (images, labels) in enumerate(trainloader):
        
        images = images.to(device)
        labels = labels.to(device)

        ## forward + backprop + loss
        images = images.float()
        logits = model(images)
        labels = labels.unsqueeze(1).float()
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()

        ## update model params
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(logits, labels, BATCH_SIZE)
    
    model.eval()
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / len(trainloader), train_acc/ len(trainloader)))


test_acc = 0.0
for i, (images, labels) in enumerate(testloader):
    images = images.to(device)
    labels = labels.to(device)
    images = images.float()
    labels = labels.unsqueeze(1).float()
    outputs = model(images)
    test_acc += get_accuracy(outputs, labels, TEST_BATCH_SIZE)

outputs = outputs.cpu()
test_pos_mask = outputs.squeeze() > 0.5
test_neg_mask = outputs.squeeze() <=0.5

plt.figure()
plt.plot(testset.test_data[test_pos_mask, 0], testset.test_data[test_pos_mask, 1], '.', color='b', markersize=1)
plt.plot(testset.test_data[test_neg_mask, 0], testset.test_data[test_neg_mask, 1], '.', color='r', markersize=1)
plt.show(block=False)
plt.close()
        
print('Test Accuracy: %.2f'%( test_acc/len(testloader)))
