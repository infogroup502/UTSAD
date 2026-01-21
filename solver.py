import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from model.RevIN import RevIN
import time
from utils.utils import *
from model.GRUSTAD import GRUSTAD
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
from model.embed import PositionalEmbedding
warnings.filterwarnings('ignore')



def norm(x):
    mean = torch.mean(x, dim=-1, keepdim=True)
    stddev = torch.std(x, dim=-1, keepdim=True)

    # 对最后一个维度进行归一化，同时添加一个小的正则项以避免除以零
    normalized_tensor = (x - mean) / (stddev + 1e-5)
    return  normalized_tensor
def minmax_norm(x):
    min, _= torch.min(x,dim=-1,keepdim=True)
    max, _ = torch.max(x,dim=-1,keepdim=True)
    return (x - min) / (max-min+1e-5)
def my_kl_loss(p, q):  # 128 1 100 100
    Min= torch.min(p)
    Min_1= torch.min(q)
    Min = torch.min(Min,Min_1)
    offset = max(-Min.item(), 0)
    res = p * (torch.log(p + 0.001+offset) - torch.log(q + 0.001+offset))
    return torch.sum(res, dim=2)  # 128 1 100->128 100


def my_kl_loss_1(p, q):  # 128 1 100 100
    res = p * (torch.log(p + 0.001) - torch.log(q + 0.001))
    return torch.mean(torch.sum(res, dim=1), dim=-1)  # 128 1 100->128 100

def normalize_vector(x):
    B,L,N = x.shape
    Sum = torch.sum(x, dim=-1).unsqueeze(-1).repeat(1,1,N)
    return x/Sum
def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def cal_similar(similar,in_size,x,num):
    if similar == "MSE":
        criterion_keep = nn.MSELoss(reduction='none')
        in_size = torch.sum(criterion_keep(in_size, x.unsqueeze(-1).repeat(1, 1, 1, num)), dim=2)
    elif similar == "cos":
        in_size = F.cosine_similarity(in_size, x.unsqueeze(-1).repeat(1, 1, 1, num), dim=2, eps=1e-8)
    elif similar == "kl":
        in_size = my_kl_loss(in_size, x.unsqueeze(-1).repeat(1, 1, 1, num))
    else:
        in_size = torch.sum(in_size, dim=2)
    return in_size
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                               win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
            self.criterion_keep = nn.L1Loss(reduction='none')
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
            self.criterion_keep= nn.MSELoss(reduction='none')

    def build_model(self):
        self.model = GRUSTAD(batch_size=self.batch_size,win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c,
                                d_model=self.d_model, local_size=self.local_size,global_size=self.global_size,
                                channel=self.input_c,mul_num=self.mul_num)


        if torch.cuda.is_available():
            self.model.cuda()
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("params_num", total_params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        op="train"
        time_now = time.time()
        train_steps = len(self.train_loader)  # 3866
        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for it, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)  #(128,100,51)
                revin_layer = RevIN(num_features=self.input_c)
                x = revin_layer(input, 'norm')
                B, L, M = x.shape
                x_local_size = []
                x_global_num = []
                x_in = []
                for index, localsize in enumerate(self.local_size):
                    num =  self.global_size[index]
                    result = []
                    front = num // 2
                    back = num - front
                    boundary = L - back


                    #in_size=0.0
                    for i in range(self.win_size):
                        if (i < front):
                            temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                            temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                            temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                            result.append(temp1)
                        elif (i > boundary):
                            temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                            temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                            result.append(temp1)
                        else:
                            temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                            result.append(temp)
                    in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)

                    site = num//2

                    num1 = localsize
                    front = num1 // 2
                    back = num1 - front

                    front1 = self.mul_num // 2
                    back1 = self.mul_num - front1
                    in_x = in_size[:, :, :, site - front:site+back]  # 去除自身的局部
                    # in_x = torch.cat((in_size[:,:,:,site-front:site],in_size[:,:,:,site+1:site+back]),dim=-1)#去除自身的局部
                    in_y = torch.cat((in_size[:,:,:,0:site-front],in_size[:,:,:,site+back:num]),dim=-1)#去除自身的全局
                    # mul_space_in = in_size[:,:,:,site-front1:site+back1]
                    x_local_size.append(in_x)
                    x_global_num.append(in_y)
                    # x_in.append(mul_space_in)
                local_reconstruction_loss_1 = 0.0
                global_reconstruction_loss_1 = 0.0
                local_reconstruction_loss_2 = 0.0
                global_reconstruction_loss_2 = 0.0
                local_contrastive_loss = 0.0
                global_contrastive_loss = 0.0
                local_1, local_2, global_1, global_2= self.model(x,x_local_size, x_global_num,op,it,x_in)

                for u in range(len(local_1)):
                    local_reconstruction_loss_1 +=self.criterion(local_1[u],x_local_size[u])
                    global_reconstruction_loss_1 += self.criterion(global_1[u], x_global_num[u])
                    local_reconstruction_loss_2 += self.criterion(local_2[u],x_local_size[u])
                    global_reconstruction_loss_2 += self.criterion(global_2[u],x_global_num[u])
                    local_contrastive_loss += self.criterion(local_1[u],local_2[u])
                    global_contrastive_loss += self.criterion(global_1[u],global_2[u])
                loss = (local_reconstruction_loss_1 + local_reconstruction_loss_2 + local_contrastive_loss) + (global_contrastive_loss + global_reconstruction_loss_1 +global_reconstruction_loss_2)
                loss.backward()

                if (it + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    if speed > 1:
                        return 1
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - it)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                self.optimizer.step()
            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        op = "test"
        # norm_op = True
        # print("norm_op:",norm_op)
        # (1) stastic on the train set
        attens_energy = []
        for it, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local_size = []
            x_global_num = []
            x_in = []
            for index, localsize in enumerate(self.local_size):
                num = self.global_size[index]
                result = []
                front = num // 2
                back = num - front
                boundary = L - back
                # in_size=0.0
                for i in range(self.win_size):
                    if (i < front):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                        result.append(temp1)
                    elif (i > boundary):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                        temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                        result.append(temp1)
                    else:
                        temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                        result.append(temp)
                in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                # in_size = cal_similar(self.similar,in_size,x,num)
                # in_size = torch.softmax(in_size,dim=-1)

                site = num // 2

                num1 = localsize
                front = num1 // 2
                back = num1 - front

                front1 = self.mul_num // 2
                back1 = self.mul_num - front1
                in_x = in_size[:, :, :, site - front:site + back]  # 去除自身的局部
                # in_x = torch.cat((in_size[:,:,:,site-front:site],in_size[:,:,:,site+1:site+back]),dim=-1)#去除自身的局部
                in_y = torch.cat((in_size[:, :, :, 0:site - front], in_size[:, :, :, site + back:num]),
                                 dim=-1)  # 去除自身的全局
                # mul_space_in = in_size[:,:,:,site-front1:site+back1]
                x_local_size.append(in_x)
                x_global_num.append(in_y)
                # x_in.append(mul_space_in)
            local_reconstruction_loss_1 = 0.0
            global_reconstruction_loss_1 = 0.0
            local_reconstruction_loss_2 = 0.0
            global_reconstruction_loss_2 = 0.0
            local_contrastive_loss = 0.0
            global_contrastive_loss = 0.0
            local_1, local_2, global_1, global_2 = self.model(x, x_local_size, x_global_num, op, it, x_in)

            for u in range(len(local_1)):
                # series_loss += torch.sum(self.criterion_keep(series[u], x_local_size[u]), dim=-1)
                # prior_loss += torch.sum(self.criterion_keep(prior[u], x_global_num[u]), dim=-1)

                local_contrastive_loss += torch.mean(torch.sum(self.criterion_keep(local_1[u],local_2[u]),dim=-1),dim=-1)
                global_contrastive_loss += torch.mean(torch.sum(self.criterion_keep(global_1[u],global_2[u]),dim=-1),dim=-1)

            metric = minmax_norm(local_contrastive_loss)*self.r + minmax_norm(global_contrastive_loss)*(1-self.r)

            metric = torch.softmax(metric, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)


        # (2) find the threshold
        attens_energy = []
        for it, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local_size = []
            x_global_num = []
            x_in = []
            for index, localsize in enumerate(self.local_size):
                num = self.global_size[index]
                result = []
                front = num // 2
                back = num - front
                boundary = L - back
                # in_size=0.0
                for i in range(self.win_size):
                    if (i < front):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                        result.append(temp1)
                    elif (i > boundary):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                        temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                        result.append(temp1)
                    else:
                        temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                        result.append(temp)
                in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                # in_size = cal_similar(self.similar,in_size,x,num)
                # in_size = torch.softmax(in_size,dim=-1)

                site = num // 2

                num1 = localsize
                front = num1 // 2
                back = num1 - front

                front1 = self.mul_num // 2
                back1 = self.mul_num - front1
                in_x = in_size[:, :, :, site - front:site + back]  # 去除自身的局部
                # in_x = torch.cat((in_size[:,:,:,site-front:site],in_size[:,:,:,site+1:site+back]),dim=-1)#去除自身的局部
                in_y = torch.cat((in_size[:, :, :, 0:site - front], in_size[:, :, :, site + back:num]),
                                 dim=-1)  # 去除自身的全局
                # mul_space_in = in_size[:,:,:,site-front1:site+back1]
                x_local_size.append(in_x)
                x_global_num.append(in_y)
                # x_in.append(mul_space_in)
            local_reconstruction_loss_1 = 0.0
            global_reconstruction_loss_1 = 0.0
            local_reconstruction_loss_2 = 0.0
            global_reconstruction_loss_2 = 0.0
            local_contrastive_loss = 0.0
            global_contrastive_loss = 0.0
            local_1, local_2, global_1, global_2 = self.model(x, x_local_size, x_global_num, op, it, x_in)

            for u in range(len(local_1)):
                # series_loss += torch.sum(self.criterion_keep(series[u], x_local_size[u]), dim=-1)
                # prior_loss += torch.sum(self.criterion_keep(prior[u], x_global_num[u]), dim=-1)

                local_contrastive_loss += torch.mean(torch.sum(self.criterion_keep(local_1[u], local_2[u]), dim=-1),
                                                     dim=-1)
                global_contrastive_loss += torch.mean(torch.sum(self.criterion_keep(global_1[u], global_2[u]), dim=-1),
                                                      dim=-1)

            metric = minmax_norm(local_contrastive_loss) * self.r + minmax_norm(global_contrastive_loss) * (1 - self.r)

            metric = torch.softmax(metric, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("anormly_ratio",self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for it, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local_size = []
            x_global_num = []
            x_in = []
            for index, localsize in enumerate(self.local_size):
                num = self.global_size[index]
                result = []
                front = num // 2
                back = num - front
                boundary = L - back
                # in_size=0.0
                for i in range(self.win_size):
                    if (i < front):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                        result.append(temp1)
                    elif (i > boundary):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                        temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                        result.append(temp1)
                    else:
                        temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                        result.append(temp)
                in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                # in_size = cal_similar(self.similar,in_size,x,num)
                # in_size = torch.softmax(in_size,dim=-1)

                site = num // 2

                num1 = localsize
                front = num1 // 2
                back = num1 - front

                front1 = self.mul_num // 2
                back1 = self.mul_num - front1
                in_x = in_size[:, :, :, site - front:site + back]  # 去除自身的局部
                # in_x = torch.cat((in_size[:,:,:,site-front:site],in_size[:,:,:,site+1:site+back]),dim=-1)#去除自身的局部
                in_y = torch.cat((in_size[:, :, :, 0:site - front], in_size[:, :, :, site + back:num]),
                                 dim=-1)  # 去除自身的全局
                # mul_space_in = in_size[:,:,:,site-front1:site+back1]
                x_local_size.append(in_x)
                x_global_num.append(in_y)
                # x_in.append(mul_space_in)
            local_reconstruction_loss_1 = 0.0
            global_reconstruction_loss_1 = 0.0
            local_reconstruction_loss_2 = 0.0
            global_reconstruction_loss_2 = 0.0
            local_contrastive_loss = 0.0
            global_contrastive_loss = 0.0
            local_1, local_2, global_1, global_2 = self.model(x, x_local_size, x_global_num, op, it, x_in)

            for u in range(len(local_1)):
                # series_loss += torch.sum(self.criterion_keep(series[u], x_local_size[u]), dim=-1)
                # prior_loss += torch.sum(self.criterion_keep(prior[u], x_global_num[u]), dim=-1)

                local_contrastive_loss += torch.mean(torch.sum(self.criterion_keep(local_1[u], local_2[u]), dim=-1),
                                                     dim=-1)
                global_contrastive_loss += torch.mean(torch.sum(self.criterion_keep(global_1[u], global_2[u]), dim=-1),
                                                      dim=-1)

            metric = minmax_norm(local_contrastive_loss) * self.r + minmax_norm(global_contrastive_loss) * (1 - self.r)

            metric = torch.softmax(metric, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        # 构造目标目录路径
        output_dir = os.path.join('pre', self.dataset)

        # 如果目录不存在，先创建它
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.savetxt(os.path.join(output_dir, 'pred.csv'), pred, fmt='%d', delimiter='\n')
        np.savetxt(os.path.join(output_dir, 'score.csv'), test_energy, fmt='%f', delimiter='\n')
        np.savetxt(os.path.join(output_dir, 'fact.csv'), gt, fmt='%d', delimiter='\n')
        np.savetxt(os.path.join(output_dir, 'discrepancy.csv'), (gt == pred).astype(int), fmt='%d', delimiter='\n')

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))

        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/' + self.data_path + '.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score
