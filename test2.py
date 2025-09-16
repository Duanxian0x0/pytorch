import torch
from torch.utils.data import Dataset # 在torch的utils的工具包中
from PIL import Image
import os


# 使用Dataset加载数据

class MyData(Dataset): # 继承Dataset
    def __init__(self,root_dir,lable_dir):
        self.root_dir = root_dir
        self.lable_dir = lable_dir
        self.path = os.path.join(self.root_dir,self.lable_dir)
        self.img_path = os.listdir(self.path) # os.listdir()列出当前所有图片的路径


    def __getitem__(self, index):
        image_name = self.img_path[index]
        image_item_path = os.path.join(self.root_dir,self.lable_dir,image_name)
        image = Image.open(image_item_path)
        label = self.lable_dir
        return image,label
    
    def __len__(self):
        return len(self.img_path)
    
if __name__ == "__main__":
    root_dir = "Data/FirstTypeData/train"
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    ants_dataset = MyData(root_dir, ants_label_dir)
    bees_dataset = MyData(root_dir, bees_label_dir)
    print(len(ants_dataset))
    print(len(bees_dataset))
    train_dataset = ants_dataset + bees_dataset # train_dataset 就是两个数据集的集合了     
    print(len(train_dataset))

    img,label = train_dataset[200]
    print("label：",label)
    img.show()