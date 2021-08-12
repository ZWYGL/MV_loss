import torch.optim as optim
from model import *
import util
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from util import StandardScaler

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, gcn_bool,
                 addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid,
                           dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)

        self.model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(),lr = lrate, weight_decay=wdecay)
        self.loss = util.masked_mae#Mean_error
        self.loss3 = self.Mean_error
        self.loss1 = self.my_softmax_entry# 分类新的loss
        # self.loss2 = self.get_softmax
        self.loss2 = self.variance
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, lambda_2, lambda_3, lambda_4, freeze, idx):
        if freeze:
            # for p in self.model.end_conv_1.parameters():
            #     p.requires_grad=False
            # for p in self.model.end_conv_2.parameters():
            #     p.requires_grad=False
            for p in self.model.filter_convs.parameters():
                p.requires_grad=False
            for p in self.model.gate_convs.parameters():
                p.requires_grad=False
            for p in self.model.residual_convs.parameters():
                p.requires_grad=False
            for p in self.model.skip_convs.parameters():
                p.requires_grad=False
            for p in self.model.bn.parameters():
                p.requires_grad=False
            for p in self.model.gconv.parameters():
                p.requires_grad=False
        classify_real_val = torch.LongTensor(np.int32(np.ceil(real_val.cpu()))).cuda()#.to("cuda:0")
        self.model.train()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        classify, predict = self.model(input, idx)
        predict_compute = self.make_predict(classify)
        loss3 = self.loss3(predict_compute,real_val)
        loss1 = self.loss1(classify, classify_real_val)
        loss2 = self.loss2(predict_compute,classify)
        loss_class = loss1 + lambda_2 * loss2 + lambda_3 * loss3 #lossMAE, LOSS1:交叉熵 LOSS2:softmax LOSS3:mean
        self.optimizer.zero_grad()
        predict1 = self.scaler.inverse_transform(predict)
        loss = self.loss(predict1,real_val,0.0)
        loss = loss * lambda_4
        loss = loss + loss_class
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mae = util.masked_mae(predict1, real_val, 0.0).item()
        mape = util.masked_mape(predict1, real_val, 0.0).item()
        rmse = util.masked_rmse(predict1, real_val, 0.0).item()

        return mae, mape, rmse

    def eval(self, input, real_val, idx):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        classify_output, predict = self.model(input, idx)
        predict = self.scaler.inverse_transform(predict)
        mae = util.masked_mae(predict, real_val, 0.0).item()
        mape = util.masked_mape(predict, real_val, 0.0).item()
        rmse = util.masked_rmse(predict, real_val, 0.0).item()
        return mae, mape, rmse

    def get_softmax(self,output,real_val):
        batch_size, dimension,sensor, time_point = output.shape
        p = torch.nn.functional.softmax(output,dim=1)
        # pos = torch.zeros(batch_size,dimension,sensor,time_point).cuda()
        # for i in range(batch_size):
        #     for j in range(sensor):
        #         try:
        #             pos[i, real_val[i][0][j][0].int(), j, 0] = 1
        #         except:
        #             pos[i, int(real_val[i][0][j][0].item()), j, 0] = 1
        # prob_gt = (pos * p).sum(1).unsqueeze(dim=1).cuda()  # probs of the gt. dim 1 is squeezed
        # # prob_gt_2 = torch.squeeze((pos * p).sum(1, keepdim=True), dim=1) # exactly the same but more verbose
        #
        # prob_gt = prob_gt.expand(batch_size, dimension, sensor, time_point)  # .t() why transpose ?
        # pos_no_K = torch.FloatTensor((p < prob_gt).float().cpu()).cuda()  # int)
        # p_not_K = pos_no_K * p

        ##batch_average_K = int(((pos_no_K == 0).sum()/(batch_size*sensor)).item())
        K = 11
        no_top_k = 71 - K
        EPS = 1e-3
        p_not_K, _ = torch.topk(p, no_top_k, dim = 1, largest=False)
        loss = (-(p_not_K + EPS) * torch.log(p_not_K + EPS)).sum() / (batch_size*sensor)
        return loss

    def my_softmax_entry(self,inputs, target):
        inputs = torch.torch.nn.functional.softmax(inputs, dim=1)
        batch_size, dimension,sensor, time_point = inputs.shape
        # 对target进行one-hot编码
        target = torch.zeros(batch_size, dimension,sensor, time_point).scatter_(1, target.cpu(), 1).to(self.device)
        # 计算交叉熵
        loss = (-((target * torch.log(inputs)).sum(3, keepdim=False)).sum(2, keepdim=False).sum(1, keepdim=False).sum(0,
                                                                                                                      keepdim=False)) / (
                           batch_size * sensor * time_point)
        return loss

    def make_predict(self, inputs):
        p = torch.nn.functional.softmax(inputs, dim=1)  # -1,72,207,1
        a = torch.arange(1, 72, dtype=torch.float32).view(1, 71, 1, 1).to(self.device)
        mean = torch.unsqueeze((p * a).sum(1), dim=1)
        return mean

    def Mean_error(self, mean, target):   #target (-1,1,207,1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0
        return mean_loss

    def variance(self, mean, inputs):
        p = torch.nn.functional.softmax(inputs, dim=1)
        a = torch.arange(0, 71, dtype = torch.float32).cuda()
        a = torch.unsqueeze(a, 0)
        a = torch.unsqueeze(a, -1)
        a = torch.unsqueeze(a, -1)
        diff = (a - mean) ** 2
        pd = p * diff
        variance_loss = pd.sum(1, keepdim = True).mean()
        return variance_loss






