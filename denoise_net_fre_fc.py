from turtle import shape
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from torchsummary import summary
from Classifier_network_base import Pertrainmodel


class Denoise_embedding(nn.Module):
    def __init__(self,freeze = True) -> None:
        super(Denoise_embedding,self).__init__()

        self.pretrain_weights_path = "Pretrain/basemodel_84_75.692.pth"
        self.Class_net = Pertrainmodel()
        # a = self.Class_net.state_dict()

        if freeze:
            self.Class_net = self.Class_net.to('cuda')
            self.Class_net.load_state_dict(torch.load(self.pretrain_weights_path))
            print("load the Classiflier model weight success")
            for p in self.Class_net.parameters():
                p.requires_grad = False
        # b = self.Class_net.state_dict()

    def forward(self, x):
        eeg_embedding = x
        noise_embedding = self.Class_net(eeg_embedding)


        return eeg_embedding, noise_embedding




class FDSconv3(nn.Module):
    def __init__(self, input_channel, input_len, reduction):
        super(FDSconv3, self).__init__()

        self.in_channel = input_channel
        self.input_len = input_len
        self.t = 0.4     #change the parameter from 0.2 to 0.9 offset 0.6

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)

        self.mlp = nn.Sequential(
            nn.Linear(10, (input_len // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((input_len // reduction), input_len, bias=False),
            nn.Sigmoid())
        self.conv = nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, padding=1)
    def forward(self, x):
        '''
        :param x[0]: eeg : B * l (512)
        :param x[1]: noise representation: B * l(10)

        '''

        emb_eeg = x[0]
        emb_noise = x[1]

        b, c, l = emb_eeg.size()
        fre_len = l // 2 + 1
        # print(fre_len)

        # return to frenquce domain
        fft = torch.fft.rfft(emb_eeg, dim=-1)
        # print(fft.shape)

        weight_init = self.mlp(emb_noise)
        # print(weight_init.shape)
        # sig_weight = self.sigmoid(weight_init).to(torch.float32)
        zero = torch.zeros(fre_len, device=weight_init.device)
        ones = torch.ones(fre_len, device=weight_init.device)
        weight1 = torch.where(weight_init > self.t, weight_init, zero)
        selectors = torch.where(weight1 > self.t, ones, weight1).view(b, 1, fre_len)
        # with open(f'selector.txt','a', encoding='utf-8') as f:
        #     for item in selectors:
        #         f.write(str(item) +';')
        #     f.close()
        

        # print(selectors[1])
        # selectors = selectors.to(torch.float32)
        # print(selectors.size)
        ifft_output = fft * selectors.expand_as(selectors)

        # return to time domain
        output = torch.fft.irfft(ifft_output, dim=-1)

        output = self.relu(self.conv(output))
        output = self.relu(self.conv(output))
        output = emb_eeg + output

        return output


#Noise_class_fusion
class NCFconv(nn.Module):
    def __init__(self,kernel_size,out_channels):
        super(NCFconv,self).__init__()
        self.kernel_size = kernel_size
        # self.in_channel = in_channels
        self.out_channel = out_channels

        self.mlp = nn.Sequential(
            nn.Linear(10,(out_channels//2)*kernel_size,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((out_channels//2)*kernel_size,out_channels*kernel_size,bias=False),#32和out_channel一致
            nn.ReLU(inplace=True))


        self.dc = nn.Conv1d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=2,padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)


    def forward(self, x):
        '''
        :param x[0]: eeg : B * l (512)
        :param x[1]: noise representation: B * l(10)

        '''

        emb_eeg = x[0]
        emb_noise = x[1]


        # emb_eeg = self.tran(emb_eeg)
        # print("1",emb_eeg.shape)
        b, c, l = emb_eeg.size()
        emb_noise = self.mlp(emb_noise)
        # print("2", emb_noise.shape)

        #(1,32*1000, 512)*(32*1000, 1, 9)=(1,32000,512)
        out = self.relu(F.conv1d(emb_eeg.view(1,-1,l), emb_noise.view(-1,1,self.kernel_size),groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.relu(self.conv(out.view(b,-1,l)))#[1000,32,512]
        out = emb_eeg + out
        # print("4", out.shape)
        # out = self.relu(self.dc(out))

        return out


class Denoise_net(nn.Module):
    def __init__(self) -> None:
        super(Denoise_net,self).__init__()


        reduction = 8
        kernel_size = 3


        self.sigle = Denoise_embedding()
        self.fds1 = FDSconv3(32, 257, reduction)
        self.fds2 = FDSconv3(64, 129, reduction)
        self.fds3 = FDSconv3(128, 65, reduction)
        self.fds4 = FDSconv3(256, 33, reduction)
        self.fds5 = FDSconv3(512, 17, reduction)
        self.fds6 = FDSconv3(1024, 9, reduction)
        self.ncf1 = NCFconv(kernel_size, 32)
        self.ncf2 = NCFconv(kernel_size, 64)
        self.ncf3 = NCFconv(kernel_size, 128)
        self.ncf4 = NCFconv(kernel_size, 256)
        self.ncf5 = NCFconv(kernel_size, 512)
        self.ncf6 = NCFconv(kernel_size, 1024)

        self.tran1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True) )


        self.dc1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.tran2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True) )
        

        self.dc2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.tran3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True) )
            

        self.dc3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)


        self.tran4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

        self.dc4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.tran5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

        self.dc5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)

        self.tran6 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

        self.dc6 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)

        self.tran_out =nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),nn.ReLU(inplace=True),
            nn.Conv1d(16, 1,  kernel_size=3, padding=1),nn.ReLU(inplace=True))

        self.flat7_1 = nn.Flatten()
        self.line7_2 = nn.Linear(8192, 512)


    def forward(self, x):
        eeg_embedding = x

        # eeg_embedding = torch.unsqueeze(eeg_embedding, 1)

        eeg_embedding, noise_embedding = self.sigle(eeg_embedding)
        # print(sigle_embedding[0].shape)

        eeg_embedding = torch.unsqueeze(eeg_embedding, 1)
        noise_embedding = torch.unsqueeze(noise_embedding, 1)

        # fre_eeg = self.freq((eeg_embedding,noise_embedding))

        # print("0",fre_eeg.shape)

        x_1 = self.tran1(eeg_embedding)#C1_l512->C32_l512
        res_1 = self.fds1((x_1,noise_embedding))
        res_1 = self.ncf1((res_1, noise_embedding))
        res_1 = res_1 + x_1
        res_1 = self.dc1(res_1)#C32_l512->C32_l256


        x_2 = self.tran2(res_1)#C32_l256->C64_l256
        res_2 = self.fds2((x_2, noise_embedding))
        res_2 = self.ncf2((res_2,noise_embedding))
        res_2 = res_2 + x_2
        res_2 = self.dc2(res_2)  # C64_l256->C64_l128

        x_3 = self.tran3(res_2)  # C64_l128->C128_l128
        res_3 = self.fds3((x_3, noise_embedding))
        res_3 = self.ncf3((res_3, noise_embedding))
        res_3 = res_3 + x_3
        res_3 = self.dc3(res_3)  # C128_l128->C128_l64

        x_4 = self.tran4(res_3)  # C128_l64->C256_l64
        res_4 = self.fds4((x_4, noise_embedding))
        res_4 = self.ncf4((res_4, noise_embedding))
        res_4 = res_4 + x_4
        res_4 = self.dc4(res_4)  # C256_l64->C256_l32

        x_5 = self.tran5(res_4)  # C256_l32->C512_l32
        res_5 = self.fds5((x_5, noise_embedding))
        res_5 = self.ncf5((res_5, noise_embedding))
        res_5 = res_5 + x_5
        res_5 = self.dc5(res_5)  # C512_l32->C512_l16

        x_6 = self.tran6(res_5)  # C512_l16->C1024_l16
        res_6 = self.fds6((x_6, noise_embedding))
        res_6 = self.ncf6((res_6, noise_embedding))
        res_6 = res_6 + x_6
        res_6 = self.dc6(res_6)  # C1024_l16->C1024_l8

        # print(group_6.shape)

        flat_7 = self.flat7_1(res_6)
        out = self.line7_2(flat_7)


        # self.flat7_1 = nn.Flatten()
        # self.line7_2 = nn.Linear(16384, 512)

        # group_5 = group_5.permute(0, 2, 1)
        # group_5 = self.tran_out(group_5)
        # print("5", group_5.shape)

        out = torch.squeeze(out, 1)
        # print(out.shape)
        return out







