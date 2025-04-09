from turtle import shape
import torch
import torch.nn as nn
import numpy as np


# import mxnet as mx
# from mxnet import autograd ,gluon,init,nd

class Pertrainmodel(nn.Module):
    def __init__(self) -> None:
        super(Pertrainmodel,self).__init__()

        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.avg_1_3 = nn.AvgPool1d(2)

        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.avg_2_3 = nn.AvgPool1d(2)

        self.conv3_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.avg_3_3 = nn.AvgPool1d(2)

        self.conv4_1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.drop4_3 = nn.Dropout(0.5)
        self.avg_4_4 = nn.AvgPool1d(2)

        self.conv5_1 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.drop5_3 = nn.Dropout(0.5)
        self.avg_5_4 = nn.AvgPool1d(2)

        self.conv6_1 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.drop6_3 = nn.Dropout(0.5)

        self.flat7_1 = nn.Flatten()
        self.line7_2 = nn.Linear(16384, 512)
        self.line7_3 = nn.Linear(512,10)

    def forward(self, x):
        emb_x = x
        # fft = torch.fft.rfft(emb_x,dim=-1)
        # emb_x = torch.fft.irfft(fft,dim=-1)


        # print(emb_x.shape)
        emb_x_1 = torch.unsqueeze(emb_x, 1)
        # print(emb_x_1.shape)

        emb_x_1 = self.conv1_1(emb_x_1)
        emb_x_1 = torch.relu(emb_x_1)
        emb_x_1 = self.conv1_2(emb_x_1)
        emb_x_1 = torch.relu(emb_x_1)
        emb_x_1 = self.avg_1_3(emb_x_1)

        emb_x_2 = self.conv2_1(emb_x_1)
        emb_x_2 = torch.relu(emb_x_2)
        emb_x_2 = self.conv2_2(emb_x_2)
        emb_x_2 = torch.relu(emb_x_2)
        emb_x_2 = self.avg_2_3(emb_x_2)

        emb_x_3 = self.conv3_1(emb_x_2)
        emb_x_3 = torch.relu(emb_x_3)
        emb_x_3 = self.conv3_2(emb_x_3)
        emb_x_3 = torch.relu(emb_x_3)
        emb_x_3 = self.avg_3_3(emb_x_3)

        emb_x_4 = self.conv4_1(emb_x_3)
        emb_x_4 = torch.relu(emb_x_4)
        emb_x_4 = self.conv4_2(emb_x_4)
        emb_x_4 = torch.relu(emb_x_4)
        emb_x_4 = self.drop4_3(emb_x_4)
        emb_x_4 = self.avg_4_4(emb_x_4)

        emb_x_5 = self.conv5_1(emb_x_4)
        emb_x_5 = torch.relu(emb_x_5)
        emb_x_5 = self.conv5_2(emb_x_5)
        emb_x_5 = torch.relu(emb_x_5)
        emb_x_5 = self.drop5_3(emb_x_5)
        emb_x_5 = self.avg_5_4(emb_x_5)

        emb_x_6 = self.conv6_1(emb_x_5)
        emb_x_6 = torch.relu(emb_x_6)
        emb_x_6 = self.conv6_2(emb_x_6)
        emb_x_6 = torch.relu(emb_x_6)
        emb_x_6 = self.drop6_3(emb_x_6)

        emb_x_7 = self.flat7_1(emb_x_6)
        emb_x_7 = self.line7_2(emb_x_7)
        emb_x_7 = torch.relu(emb_x_7)

        output = self.line7_3(emb_x_7)
        output = torch.squeeze(output, 1)

        return output