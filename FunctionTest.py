"""模型测试统一入口。

python FunctionTest.py                     # baseline 和 decoupling 都运行
python FunctionTest.py --model baseline    # 只测试原 baseline
python FunctionTest.py --model decoupling  # 只测试解耦融合模型
"""

import argparse

import torch
import torch.nn.functional as F
from torch import nn

from model import *
from model_decoupling import Net as DecouplingNet, cal_decomp_loss

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


def model_decoupling_fun():
    """从 model_decoupling.py 迁入的前向、反向和全局残差自检。

    自动选择可用的 GPU 或 CPU；不加载数据，不保存或覆盖模型权重。
    """
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecouplingNet(31, 3).to(device)
    lrhs = torch.rand(2, 31, 8, 8, device=device)
    hrms = torch.rand(2, 3, 64, 64, device=device)
    target = torch.rand(2, 31, 64, 64, device=device)
    prediction, aux = model(lrhs, hrms, return_aux=True)
    assert prediction.shape == target.shape
    assert all(x.shape == (2, 64, 64, 64) for x in aux.values())
    loss_sc = cal_decomp_loss(
        aux["msi_shared"], aux["msi_complementary"], aux["hsi_shared"], aux["hsi_complementary"],
    )
    loss = F.l1_loss(prediction, target) + 0.1 * loss_sc
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        model.tail.weight.zero_()
        model.tail.bias.zero_()
        torch.testing.assert_close(
            model(lrhs, hrms), F.interpolate(lrhs, size=(64, 64), mode="bilinear", align_corners=False),
            rtol=0, atol=0,
        )
    print("Device:", device)
    print("HR-HSI shape:", tuple(prediction.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("L_SC:", float(loss_sc.detach()))
    print("Forward, backward and global residual checks passed.")


def _main():
    parser = argparse.ArgumentParser(description="Run baseline and/or decoupling model checks")
    parser.add_argument("--model", choices=("baseline", "decoupling", "all"), default="all")
    args = parser.parse_args()
    if args.model in ("baseline", "all"):
        model_fun()
    if args.model in ("decoupling", "all"):
        model_decoupling_fun()


if __name__ == '__main__':
    _main()
