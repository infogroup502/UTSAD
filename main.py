import os
import argparse
import numpy as np
from torch.backends import cudnn
from utils.utils import *
from solver import Solver
import time
import warnings

warnings.filterwarnings('ignore')

import sys


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 示例用法
set_seed(42)


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def str2bool(v):
    return v.lower() in ('true')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(array[idx - 1])


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    # solver = Solver(vars(config))

    # win_size = [10,20,30,40,50,60,90,105,70,80,100,110,130,150]
    # # win_size = [ 70, 80, 100, 110, 130, 150]
    # local_size = [2,3,5,7,9]
    # global_size = [10,15,20,25,30]
    # # win_size = [30]#20 3 10    20 3 15  20 5 10   20 7 10  20 9 15  20 9 20
    # # local_size = [9]
    # # global_size = [10]
    # for win in win_size:
    #     for local in local_size:
    #         for global_ in global_size:
    #             if global_ > win :
    #                 continue
    #             set_seed(42)
    #             print("win_size:{}, local_size:{}, global_size:{}".format(win, local, global_))
    #             config.win_size = win
    #             config.local_size = [local]
    #             config.global_size = [global_]
    #             solver = Solver(vars(config))
    #             x =solver.train()
    #             if x==1:
    #                 continue
    #             # elif config.mode == 'test':
    #             solver.test()
    # if config.mode == 'train':

    solver = Solver(vars(config))
    solver.train()
    # elif config.mode == 'test':
    test_time = time.time()
    solver.test()
    print("test time:", time.time() - test_time)
    return solver
import sys
import time
import psutil
import GPUtil
from threading import Thread
class Monitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.cpu_usage = []
        self.mem_usage = []
        self.gpu_usage = []
        self.running = False

    def start(self):
        self.running = True
        self.thread = Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _monitor(self):
        while self.running:
            self.cpu_usage.append(psutil.cpu_percent())
            self.mem_usage.append(psutil.virtual_memory().percent)
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_usage.append(sum([gpu.load * 100 for gpu in gpus]) / len(gpus))
            time.sleep(self.interval)

    def get_avg_usage(self):
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_mem = sum(self.mem_usage) / len(self.mem_usage) if self.mem_usage else 0
        avg_gpu = sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0
        return avg_cpu, avg_mem, avg_gpu

monitor = Monitor(interval=1)

import subprocess

def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    memory_used = int(result.stdout.strip()) / 1024
    return memory_used



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    monitor.start()
    def list_type(arg):
        return [int(item) for item in arg.split(',')]
    # Alternative
    parser.add_argument('--win_size', type=int, default=50)
    parser.add_argument('--local_size', type=list_type, default=[9])#时域局部大小
    parser.add_argument('--global_size', type=list_type, default=[20])#时域全局
    parser.add_argument('--mul_num', type=int, default=10) #Less than or equal to global_size 空间全局大小
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=64)  ######## 隐藏维度
    parser.add_argument('--anormly_ratio', type=float, default=0.4)#
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)  #######
    parser.add_argument('--r', type=float, default=0.5)

    

    parser.add_argument('--rec_timeseries', action='store_true', default=True)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--loss_fuc', type=str, default='MSE')
    # Default
    parser.add_argument('--index', type=int, default=137)

    parser.add_argument('--input_c', type=int, default=8)  ##########MSL 55
    parser.add_argument('--output_c', type=int, default=8)  #########
    parser.add_argument('--dataset', type=str, default='SKAB')
    parser.add_argument('--data_path', type=str, default='SKAB')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')



    config = parser.parse_args()
    args = vars(config)
    config.local_size = [int(patch_index) for patch_index in config.local_size]

    if config.dataset == 'UCR':
        batch_size_buffer = [2, 4, 8, 16, 32, 64, 128, 256]
        data_len = np.load('dataset/' + config.data_path + "/UCR_" + str(config.index) + "_train.npy").shape[0]
        config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)
    elif config.dataset == 'UCR_AUG':
        batch_size_buffer = [2, 4, 8, 16, 32, 64, 128, 256]
        data_len = np.load('dataset/' + config.data_path + "/UCR_AUG_" + str(config.index) + "_train.npy").shape[0]
        config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)
    elif config.dataset == 'SMD_Ori':
        batch_size_buffer = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        data_len = np.load('dataset/' + config.data_path + "/SMD_Ori_" + str(config.index) + "_train.npy").shape[0]
        config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)

    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ', '')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]

    sys.stdout = Logger("result/" + config.data_path + ".log", sys.stdout)
    print("\n\n")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('================ Hyperparameters ===============')
    print('dataset:', config.data_path)
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('====================  Train  ===================')

    main(config)
    monitor.stop()

    avg_cpu, avg_mem, avg_gpu = monitor.get_avg_usage()

    print(f"Average CPU Usage: {avg_cpu:.1f}%")
    print(f"Average Memory Usage: {avg_mem:.1f}%")
    print(f"Average GPU Usage: {avg_gpu:.1f}%")


