import torch

def check_cuda():
    print("cuda可用性{0}".format(torch.cuda.is_available()))

if __name__ == "__main__":
    check_cuda()


