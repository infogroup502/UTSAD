import torch
import torch.nn as nn
from einops import rearrange
from model.RevIN import RevIN
from tkinter import _flatten
import torch.nn.functional as F
import math
from model.kan import KAN,KANLinear
from model.embed import PositionalEmbedding


class Conv1d(nn.Module):
    def __init__(self, input_size,  output_size):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(input_size,output_size,3,1,1)#kernel_size3卷积核大小 stride1卷积步长 padding1填充数量


    def forward(self, x):
        x = torch.relu(self.conv(x))  # 使用ReLU作为激活函数
        return x

# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size,bias=False)  # 第一个全连接层
#         self.fc2 = nn.Linear(hidden_size, output_size,bias=False) # 第二个全连接层
#         self.b_spline_make = KANLinear(input_size, hidden_size,  grid_size=3, spline_order=1)
#
#     def forward(self, x):
#         # silu = torch.nn.SiLU()
#         x = torch.relu(self.fc1(x))+self.b_spline_make(x)  # 使用ReLU作为激活函数
#         x = self.fc2(x)
#         return x
# class RBF(nn.Module):
#     def __init__(self,B,L,M , input_size, hidden_size, output_size,sigma=0.1):
#         super(RBF, self).__init__()
#         self.sigma = sigma
#         self.hidden_size = hidden_size
#         self.centers = nn.Parameter(torch.randn(L, hidden_size))
#         self.weight = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         B, L, M,_ = x.shape
#         centers = self.centers.reshape(1,L,1,self.hidden_size).repeat(B, 1, M, 1)
#         distance = x.unsqueeze(-2) - centers.unsqueeze(-1)
#         x = torch.sum(torch.exp(-distance**2/(2*self.sigma**2)),dim=-1)
#         x = self.weight(x)
#         return x


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
#         self.fc2 = nn.Linear(hidden_size, output_size) # 第二个全连接层
#
#     def forward(self, x):
#         silu = torch.nn.SiLU()
#         x = silu(self.fc1(x))  # 使用ReLU作为激活函数
#         x = self.fc2(x)
#         return x
#
# def calculate_area(mu, sigma, x):
#     # 计算正态分布的累积分布函数 CDF
#     dist = torch.distributions.Normal(mu, sigma)
#     cdf1 = dist.cdf(x[:,:,0])
#     cdf2 = dist.cdf(x[:,:,-1])
#
#     return cdf1 - cdf2
#
# def calculate_area_1(mu, sigma, x):
#     # 计算正态分布的累积分布函数 CDF
#     dist = torch.distributions.Normal(mu, sigma)
#     cdf = dist.cdf(x[:,:,-1])
#
#     return -cdf
class GRUSTAD(nn.Module):
    def __init__(self, batch_size,win_size, enc_in, c_out, d_model=256, local_size=[3], global_size=[1], channel=55, dropout=0.05,mul_num=3, output_attention=True, ):
        super(GRUSTAD, self).__init__()
        self.output_attention = output_attention
        self.local_size = local_size
        self.global_size = global_size
        self.channel = channel
        self.win_size = win_size
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.batch_size = batch_size
        self.mul_num=mul_num
        self.mlp_local_1 = nn.ModuleList(
            Conv1d(localsize ,global_size[index] - localsize) for index, localsize in enumerate(self.local_size))
        self.mlp_global_1 = nn.ModuleList(
            Conv1d(global_size[index] - localsize, localsize) for index, localsize in enumerate(self.local_size))
        self.mlp_local_2 = nn.ModuleList(
            Conv1d(localsize ,  global_size[index] - localsize) for index, localsize in enumerate(self.local_size))
        self.mlp_global_2 = nn.ModuleList(
            Conv1d(global_size[index] - localsize,localsize) for index, localsize in enumerate(self.local_size))
        # self.rbf_local_1 = nn.ModuleList(
        #     RBF(batch_size, win_size, channel, localsize, d_model, global_size[index] - localsize) for index, localsize in enumerate(self.local_size))
        # self.rbf_global_1 = nn.ModuleList(
        #     RBF(batch_size, win_size, channel,global_size[index] - localsize, d_model, localsize) for index, localsize in enumerate(self.local_size))
        # self.rbf_local_2 = nn.ModuleList(
        #     RBF(batch_size, win_size, channel,localsize, d_model, global_size[index] - localsize) for index, localsize in enumerate(self.local_size))
        # self.rbf_global_2 = nn.ModuleList(
        #     RBF(batch_size, win_size, channel,global_size[index] - localsize, d_model, localsize) for index, localsize in enumerate(self.local_size))

        # self.kan_local_time = nn.ModuleList(
        #     KAN([localsize-1, d_model,1 ]) for index, localsize in enumerate(self.local_size))
        # self.kan_global_time = nn.ModuleList(
        #     KAN([global_size[index]-localsize, d_model, 1]) for index, localsize in enumerate(self.local_size))
        # self.kan_local_space = nn.ModuleList(
        #     KAN([channel, d_model, channel]) for index, localsize in enumerate(self.local_size))
        # self.kan_global_space = nn.ModuleList(
        #     KAN([channel, d_model, channel]) for index, localsize in enumerate(self.local_size))
        # self.position_embedding = PositionalEmbedding(d_model=d_model)
    def forward(self,x_in, in_size, in_num, op, it,in_x):
        local_1 = []
        global_1 = []
        local_2 = []
        global_2 = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        B, L, M = x_in.shape

        for index, localsize in enumerate(self.local_size):
            input = in_size[index].permute(0,2,3,1).reshape(B*M,-1,L)
            input = self.mlp_local_1[index](input)#local->global
            global_1.append(input.reshape(B,M,-1,L).permute(0,3,1,2))
            input = self.mlp_global_1[index](input)#global->local
            local_1.append(input.reshape(B,M,-1,L).permute(0,3,1,2))
            input = self.mlp_local_2[index](input)#local->global
            global_2.append(input.reshape(B,M,-1,L).permute(0,3,1,2))
            input = self.mlp_global_2[index](input)#global->local
            local_2.append(input.reshape(B,M,-1,L).permute(0,3,1,2))

            # local_out_time.append(self.kan_local_time[index](in_size[index]).reshape(B,L,M))
            # global_out_time.append(self.kan_global_time[index](in_num[index]).reshape(B,L,M))
            # local_out_space.append(self.kan_local_space[index](x_in).reshape(B,L,M))
            # global_out_space.append(torch.mean(
            #     self.kan_global_space[index](in_x[index].permute(0,1,3,2)).permute(0,1,3,2),dim=-1))

        local_1 = list(_flatten(local_1))  # 3
        global_1= list(_flatten(global_1))  # 3
        local_2 = list(_flatten(local_2))  # 3
        global_2 = list(_flatten(global_2))  # 3

        if self.output_attention:
            return local_1, local_2, global_1, global_2
        else:
            return None



