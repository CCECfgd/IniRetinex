import torchvision
from torchvision import transforms
from PIL import Image
import torch
from torch.fft import fft2, ifft2
import torch.nn.functional as F


def multiplyd(T):

    Dx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=T.dtype, device=T.device).view(1, 1, 3, 3)
    Dy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=T.dtype, device=T.device).view(1, 1, 3, 3)

    Dx = Dx.repeat(T.shape[1], 1, 1, 1)
    Dy = Dy.repeat(T.shape[1], 1, 1, 1)

    delTx = F.conv2d(T, Dx, padding=1, groups=T.shape[1])
    delTy = F.conv2d(T, Dy, padding=1, groups=T.shape[1])

    delT = torch.cat((delTx, delTy), dim=2)  # Concatenate along the height dimension
    return delT


def make_weight_matrix(Ti, ker_size):
    b, c, h, w = Ti.shape
    delTi = multiplyd(Ti)
    dtx = delTi[:, :, :h, :]
    dty = delTi[:, :, h:, :]
    g_ker_radius = ker_size // 2
    g_ker_range = torch.arange(-g_ker_radius, g_ker_radius + 1, dtype=Ti.dtype, device=Ti.device)
    w_gauss = torch.exp(-0.5 * (g_ker_range / 2) ** 2)
    w_gauss = w_gauss / w_gauss.sum()
    w_gauss = w_gauss.view(1, 1, 1, -1)
    convl_x = F.conv2d(dtx, w_gauss, padding=(0, g_ker_radius), groups=c)
    convl_y = F.conv2d(dty, w_gauss, padding=(0, g_ker_radius), groups=c)
    W_x = 1 / (convl_x.abs() + 0.0001)
    W_y = 1 / (convl_y.abs() + 0.0001)
    return W_x,W_y

def total_variation1(S, h_input, v_input, lambd, device='cuda'):

    MAX_ITER = 7
    rho = 15  # 增广系数
    beta = 1.8  # 1.5-1.8 超松弛参数，用来加快收敛速度
    gamma = 1  # *0.5  # 稀疏惩罚系数，越大越平滑


    # 确保输入张量在正确的设备上
    S = S.to(device)
    h_input = h_input.to(device)
    v_input = v_input.to(device)

    batch_size, channels, height, width = S.shape
    w = gamma / rho  # 软阈值

    Wx,Wy = make_weight_matrix(S, 2)

    x = torch.zeros_like(S)
    z1 = torch.zeros_like(S)
    z2 = torch.zeros_like(S)
    u1 = torch.zeros_like(S)
    u2 = torch.zeros_like(S)

    fx = torch.tensor([1, -1], dtype=torch.float, device=device).view(1, 1, 1, 2)
    fy = torch.tensor([1, -1], dtype=torch.float, device=device).view(1, 1, 2, 1)

    otfFx = fft2(fx, s=(height, width))
    otfFy = fft2(fy, s=(height, width))


    # 扩展以匹配批次和通道
    otfFx = otfFx.expand(batch_size, channels, -1, -1)
    otfFy = otfFy.expand(batch_size, channels, -1, -1)

    for _ in range(MAX_ITER):

        h = lambd * h_input + rho / 2 * (z1 - u1)
        v = lambd * v_input + rho / 2 * (z2 - u2)
        h_diff_pad1 = torch.cat((h[:, :, :, -1:], -torch.diff(h, dim=3)), dim=3)
        v_diff_pad1 = torch.cat((v[:, :, -1:, :], -torch.diff(v, dim=2)), dim=2)
        Normin11 = h_diff_pad1 + v_diff_pad1
        FS = (fft2(S) + fft2(Normin11)) / (1 + (lambd + rho / 2) * (abs(otfFx) ** 2 + abs(otfFy) ** 2))

        x = ifft2(FS).real

        x_diff_h = torch.cat((torch.diff(x, dim=3), x[:, :, :, :1] - x[:, :, :, -1:]), dim=3)
        x_diff_v = torch.cat((torch.diff(x, dim=2), x[:, :, :1, :] - x[:, :, -1:, :]), dim=2)

        # warm start，这里使用了超松弛技巧来提高收敛速度
        Ax_hat1 = beta * x_diff_h + (1 - beta) * z1
        Ax_hat2 = beta * x_diff_v + (1 - beta) * z2
        # 软阈值算子
        # temp1 = abs(Ax_hat1+u1) - w*Wx
        z1 = torch.max(abs(Ax_hat1+u1) - w*Wx, torch.zeros_like(Ax_hat1)) * Ax_hat1.sign()
        z2 = torch.max(abs(Ax_hat2+u2) - w*Wy, torch.zeros_like(Ax_hat2)) * Ax_hat2.sign()
        # 更新对偶变量：u = u + z - x_diff
        u1 = u1 + Ax_hat1 - z1
        u2 = u2 + Ax_hat2 - z2

    return x


# 模拟图像数据
def smooth(S, lambd=7):#7
    # beta = beta
    padnum = 20
    S = torch.nn.ReflectionPad2d(padnum)(S)  # padding
    # 计算 h_input 和 v_input
    h_input = torch.cat((torch.diff(S, dim=3), S[:, :, :, :1] - S[:, :, :, -1:]), dim=3)
    v_input = torch.cat((torch.diff(S, dim=2), S[:, :, :1, :] - S[:, :, -1:, :]), dim=2)
    # 调用 total_variation1 函数
    x = total_variation1(S, h_input, v_input, lambd)
    x = x[:, :, padnum:-padnum, padnum:-padnum]  # remove padding
    return x


