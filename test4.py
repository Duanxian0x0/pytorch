# 使用torchvision数据

import torchvision
from torch.utils.data import DataLoader


# 加载数据
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True)

print(test_set[0])
print(test_set.classes)

img,target = test_set[0]
print(img)
print(target)

# 使用dataloader
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTenso())
test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)

for data in test_loader:
    imgs,targets = data
    print(imgs)
    print(targets)

