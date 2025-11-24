import torch.nn as nn
from smooth_RTV import *

class Network(nn.Module):
    def __init__(self,gamma,down,denoise = False,):
        super(Network, self).__init__()
        self.denoise = denoise
        self.gamma = gamma
        channal = 8
        self.res_conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, channal, 3, 1, 0),
            nn.GroupNorm(num_channels=channal, num_groups=int(channal/2), affine=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channal, 1, 3, 1, 0),
            nn.Sigmoid()
            )
        self.down2 = nn.Upsample(scale_factor=0.5,mode='bilinear', align_corners=True)#0.5 0.25
        self.up2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)#0.5*4 0.25 *16
        self.down8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)  # 0.5 0.25
        self.loss = nn.MSELoss()
        self.maxchannal = nn.MaxPool3d((3,1,1))
        if self.denoise:
            from model.DANet.networks.UNetD import UNetD
            self.denoiser = UNetD(3, wf=32, depth=5).cuda()
            self.denoiser.load_state_dict(torch.load('./model/DANet/DANetPlus.pt', map_location='cpu'))
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)

    def forward(self, input):
        maxc,_ = torch.max(input,1)
        I = self.res_conv1(input)
        Imap_ = I + maxc
        return Imap_,I
    def test(self, input):

        inp = self.down2(input)
        maxc, _ = torch.max(inp, 1)
        maxc = maxc.unsqueeze(0)
        I = self.res_conv1(inp)
        Imap_ = maxc+I
        Imap = smooth(Imap_, )#*1.5
        Imap = torch.clamp(self.up2(Imap),0.0001,1)
        R = input / Imap
        if self.denoise:
            R = self._denoise(R)
        return I, R, Imap,self.up2(maxc)
    def _loss(self, input):
        input = self.down8(input)
        I,a = self(input)
        loss = 1*self.loss(input,I)
        return loss, I
