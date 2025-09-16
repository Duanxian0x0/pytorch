# 使用nn.module,此为神经网络的基本类
import torch
from torch import nn


class testModel(nn.Module):
    def __init__(self):
        super(testModel,self).__init__() # 找到父类的初始化方法

    def forward(self,input):
        output = input+1
        return output

if __name__ == "__main__":
    new_model = testModel()
    x = torch.tensor(1.0)
    output = new_model(x)
    print(output)