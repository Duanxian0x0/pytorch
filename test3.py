# 使用transforms，理解为集成工具箱

from torchvision import transforms
from PIL import Image


def trans2tensor(image_path):
    image = Image.open(image_path)

    # 对图片数据进行清理

    tensor_trans = transforms.ToTensor() # 创建tensor器实例化对象
    tensor_image = tensor_trans(image)
    return tensor_image

# 到此图片数据就被转化为tensor格式了
# 为什么需要tensor格式的数据
# 为了反向传播，梯度等属性，包括了神经网络需要的一些属性

# 对于Normanize进行归一化

def tensor2Norm(tensor_image):
    # 创建一个Norm的归一化器
    tensor_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #input[channel]=(input[chnnel]-mean[channel])/std[channel]            
    img_norm = tensor_norm(tensor_image)  
    # 创建一个resize器
    reszie_image = transforms.Resize((512,512))
    # Resize 第二种方式：等比缩放
    trans_resize_2 = transforms.Resize(512) # 512/464 = 1.103 551/500 = 1.102
    # PIL类型的 Image -> resize -> PIL类型的 Image -> totensor -> tensor类型的 Image
    trans_compose = transforms.Compose([trans_resize_2, tensor_image]) # Compose函数中后面一个参数的输入为前面一个参数的输出   
