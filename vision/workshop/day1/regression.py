import torch
import torch.nn as nn


model = nn.Linear(in_features=1, out_features=1, bias=False)

x = torch.FloatTensor([1])
y = torch.FloatTensor([0])

model.weight.data = torch.FloatTensor([0.8])


optimizer = torch.optim.SGD(params=model.parameters(), lr=0.025)


for num_iter in range(30):
    pred = model(x)

    loss = torch.pow(y-pred,2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    print('iter : %d, weight : %f'%(num_iter,model.weight.data.float()))