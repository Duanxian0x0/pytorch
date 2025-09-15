import torch

def check_cuda():
    print("cuda可用性{0}".format(torch.cuda.is_available()))

if __name__ == "__main__":
    check_cuda()


# ❯ /home/duanxian0x0/.conda/envs/myenv/bin/python /home/duanxian0x0/桌面/github/test1.py
# cuda可用性False