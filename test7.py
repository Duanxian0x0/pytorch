import torch
import torchvision
from torch import nn 
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)       
dataloader = DataLoader(dataset, batch_size=64,drop_last=True)

class mytest(nn.Module):
    def __init__(self):
        super(mytest,self).__init__()
        self.model1 = Sequential(
        Conv2d(3,32,5,padding=2),
        MaxPool2d(2),
        Conv2d(32,32,5,padding=2),
        MaxPool2d(2),
        Conv2d(32,64,5,padding=2),
        MaxPool2d(2),
        Flatten(),
        Linear(1024,64),
        Linear(64,10)
    )
        
    def forward(self, x):
        x = self.model1(x)
        return x
    

loss = nn.CrossEntropyLoss()
test = mytest()
optim = torch.optim.SGD(mytest.parameters(),lr=0.01)



for data in dataloader:
    imgs,targets = data
    output = mytest(imgs)
    result_loss = loss(output,targets)
    optim.zero_grad()
    result_loss.backward()
    optim.step()
