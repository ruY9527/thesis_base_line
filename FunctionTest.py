import torch
import torch.nn.functional as F
from torch import nn

from model import *

'''
    该文件仅仅用方法的测试
'''

def calculate_metrics_fun():
    SAM=Loss_SAM()
    RMSE=Loss_RMSE()
    PSNR=Loss_PSNR()
    psnr_list=[]
    sam_list=[]
    sam=AverageMeter()
    rmse=AverageMeter()
    psnr=AverageMeter()
    path = 'D:\LYY\YJX_fusion\model_save\\fusion_model_v9_1\cavee_test/'
    imglist = os.listdir(path)

    for i in range(0, len(imglist)):
        img = loadmat(path + imglist[i])
        lable = img["rea"]
        recon = img["fak"]
        sam_temp=SAM(lable,recon)
        psnr_temp=PSNR(torch.Tensor(lable), torch.Tensor(recon))
        sam.update(sam_temp)
        rmse.update(RMSE(torch.Tensor(lable),torch.Tensor(recon)))
        psnr.update(psnr_temp)
        psnr_list.append(psnr_temp)
        sam_list.append(sam_temp)
    print(sam.avg)
    print(rmse.avg)
    print(psnr.avg)
    print(psnr_list)
    print(sam_list)

def model_fun():
    # 自检：输入形状与 train_cave_M.py 中 profile 的调用保持一致
    lrhs = torch.randn(2, 31, 8, 8)
    hrms = torch.randn(2, 3, 64, 64)
    net = Net(31, 3)
    out = net(lrhs, hrms)
    print('输入 LR-HSI:', tuple(lrhs.shape), ' HR-MSI:', tuple(hrms.shape))
    print('输出 HR-HSI:', tuple(out.shape))
    print('参数量: %.3fM' % (sum(p.numel() for p in net.parameters()) / 1e6))


if __name__ == '__main__':
    model_fun()