import os
import torch.nn as nn
from tqdm import tqdm
from block import fusions
from evaluate import evaluate
from embedder import embedder
import numpy as np
import random as random
import torch
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)



class SSMGRL(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.criteria = nn.BCEWithLogitsLoss()
        self.cfg = args.cfg
        self.sigm = nn.Sigmoid()
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # # ===================================================#
        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        print("Started training...")
        model = trainer(self.args)
        model = model.to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        model.train()

        #for epoch in tqdm(range(self.args.nb_epochs+1)):
        for epoch in range(self.args.nb_epochs + 1):
            optimiser.zero_grad()
            loss = model(features, adj_list)

            loss.backward()
            optimiser.step()
        # torch.save(model.state_dict(), 'saved_model/best_{}_{}.pkl'.format(self.args.dataset,self.args.custom_key))
        # if self.args.use_pretrain:
        #     model.load_state_dict(torch.load('saved_model/best_{}_{}.pkl'.format(self.args.dataset,self.args.custom_key)))
        print('loss', loss)
        print("Evaluating...")
        model.eval()
        hf = model.embed(features, adj_list)
        macro_f1s, micro_f1s, k1, st = evaluate(hf, self.idx_train, self.idx_val, self.idx_test, self.labels,task=self.args.custom_key,epoch = self.args.test_epo,lr = self.args.test_lr,iterater=self.args.iterater,losstype=self.args.losstype) #,seed=seed


        #调参
        filePath = "log"
        exp_ID = 0
        for filename in os.listdir(filePath):
            file_info = filename.split("_")
            file_dataname = file_info[0]
            if file_dataname == self.args.dataset:
                exp_ID = max(int(file_info[1]), exp_ID)
        exp_name = self.args.dataset + "_" + str(exp_ID + 1)
        exp_name = os.path.join(filePath, exp_name)
        os.makedirs(exp_name)
        arg_file = open(os.path.join(exp_name, "arg.txt"), "a")
        for k, v in sorted(self.args.__dict__.items()):
            arg_file.write("\n- {}: {}".format(k, v))
        os.rename(exp_name, exp_name + "_" + '%.4f' % np.mean(macro_f1s) + "+_" + '%.4f' % np.std(
            macro_f1s) + "_" + '%.4f' % np.mean(micro_f1s) + "+_" + '%.4f' % np.std(micro_f1s))
        arg_file.write(
            "\n- macro_f1s:{},std:{}, micro_f1s:{},std:{},k1:{},std:{}".format(np.mean(macro_f1s), np.std(macro_f1s),
                                                                               np.mean(micro_f1s), np.std(micro_f1s),
                                                                               np.mean(k1), np.std(k1)))
        arg_file.close()

        return macro_f1s, micro_f1s, k1, st




def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class trainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cfg = args.cfg
        self.MLP1 = make_mlplayers(args.ft_size, args.cfg)
        self.MLP2 = make_mlplayers(args.ft_size, args.cfg)
        self.MLP3 = make_mlplayers(args.ft_size, args.cfg)
        length = args.length
        self.w_list = nn.ModuleList([nn.Linear(cfg[-1], cfg[-1], bias=True) for _ in range(length)])
        self.y_list = nn.ModuleList([nn.Linear(cfg[-1], 1) for _ in range(length)])
        self.W = nn.Parameter(torch.zeros(size=(length * cfg[-1], cfg[-1])))
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)
        self.whiten_net = WhitenTran()
        self.whiten_bn = nn.BatchNorm1d(cfg[-1], affine=False)



    def combine_att(self, h_list):
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h

    def standardization(data, dim=-1):
        # N * d
        mu = torch.mean(data, dim=dim, keepdim=True)
        sigma = torch.std(data, dim=dim, keepdim=True)
        return (data - mu) / (sigma + 1e-4)
    def ins_loss(self,z1, z2):
        whiten_net = WhitenTran()
        # z1 = self.standardization(z1)
        # z2 = self.standardization(z2)
        z1 = whiten_net.zca_forward(z1.transpose(0, 1))  # d * N
        z2 = whiten_net.zca_forward(z2.transpose(0, 1))
        c = torch.mm(z1.transpose(0, 1), z2)  # N * N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        loss = on_diag
        return loss

    def fea_loss(self,z1, z2):
        whiten_net = WhitenTran()
        # z1 = self.standardization(z1, dim=0)
        # z2 = self.standardization(z2, dim=0)
        z1 = whiten_net.zca_forward(z1)  # N * d
        z2 = whiten_net.zca_forward(z2)
        c = torch.mm(z1.transpose(0, 1), z2)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        loss = on_diag
        return loss
    def whitened_barlow_twins_loss(self, z1, z2, num):
        # z1 = self.whiten_bn(z1)
        # z2 = self.whiten_bn(z2)

        z1 = self.whiten_net.zca_forward(z1)    # N * d
        z2 = self.whiten_net.zca_forward(z2)

        c = torch.mm(z1.transpose(0, 1), z2)
        # c.div_(self.args.batch_size)
        loss = torch.diagonal(c).add_(-1).pow_(2).sum()
        #是否使用off_diag
        if self.args.diagoff:
            off_diag = off_diagonal(c).pow_(2).sum()
            if(num==1):
                loss += self.args.lambd0 * off_diag
            elif(num==2):
                loss += self.args.lambd1 * off_diag
            elif (num == 3):
                loss += self.args.lambd2 * off_diag
            elif (num == 4):
                loss += self.args.lambd3 * off_diag
            elif (num == 5):
                loss += self.args.lambd4 * off_diag
            elif (num == 6):
                loss += self.args.lambd5 * off_diag
        return loss

    def forward(self, x, adj_list=None,epoch=0,encode_time=[]):
        x = F.dropout(x, self.args.dropout, training=self.training)
        if self.args.length == 2:
            h_a = self.MLP1(x)
            h_a_1 = self.MLP2(x)

        elif self.args.length == 3:

            h_a = self.MLP1(x)
            h_a_1 = self.MLP2(x)
            h_a_2 = self.MLP3(x)

        h_p_list = []
        i = 0
        for adj in adj_list:
            if self.args.sparse:
                if i == 0 :
                    h_p = torch.spmm(adj, h_a)
                elif i == 1:
                    h_p = torch.spmm(adj, h_a_1)
                elif i == 2:
                    h_p = torch.spmm(adj, h_a_2)
                h_p_list.append(h_p)
                # h_p_list_3.append(h_p_3)
            else:
                h_p = torch.mm(adj, h_a)
                h_p_list.append(h_p)
            i += 1
        if self.args.length == 2:
            if self.args.losstype == 'nowhite':
                loss_0=self.whitened_barlow_twins_loss(h_p_list[0], h_p_list[1], 1)
                loss_1 = self.whitened_barlow_twins_loss(h_p_list[0], h_a, 2)
                loss_2 = self.whitened_barlow_twins_loss(h_p_list[1], h_a_1, 3)

                loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2
            elif self.args.fea_ins==True:
                loss_0 = self.ins_loss(h_p_list[0], h_p_list[1])+self.fea_loss(h_p_list[0], h_p_list[1])
                loss_1 = self.ins_loss(h_p_list[0], h_a)+self.fea_loss(h_p_list[0], h_a)
                loss_2 = self.ins_loss(h_p_list[1], h_a_1)+self.fea_loss(h_p_list[1], h_a_1)

                loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2
            elif self.args.fea==True:
                loss_0 = self.fea_loss(h_p_list[0], h_p_list[1])
                loss_1 = self.fea_loss(h_p_list[0], h_a)
                loss_2 = self.fea_loss(h_p_list[1], h_a_1)
                loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2
            elif self.args.ins==True:
                loss_0 = self.ins_loss(h_p_list[0], h_p_list[1])
                loss_1 = self.ins_loss(h_p_list[0], h_a)
                loss_2 = self.ins_loss(h_p_list[1], h_a_1)
                loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2
            else:
                c = (h_p_list[1]).T @ (h_p_list[0])
                c_0 = (h_p_list[0]).T @ (h_a)
                c_1 = (h_p_list[1]).T @ (h_a_1)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                loss_0 = on_diag + self.args.lambd0 * off_diag

                on_diag_0 = torch.diagonal(c_0).add_(-1).pow_(2).sum()
                off_diag_0 = off_diagonal(c_0).pow_(2).sum()
                loss_1 = on_diag_0 + self.args.lambd1 * off_diag_0
                #
                on_diag_1 = torch.diagonal(c_1).add_(-1).pow_(2).sum()
                off_diag_1 = off_diagonal(c_1).pow_(2).sum()
                loss_2 = on_diag_1 + self.args.lambd2 * off_diag_1
                loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2
        elif self.args.length == 3:
            if self.args.losstype == 'nowhite':
                loss_0 = self.whitened_barlow_twins_loss(h_p_list[0], h_p_list[1],1)
                loss_1 = self.whitened_barlow_twins_loss(h_p_list[1], h_p_list[2],2)
                loss_2 = self.whitened_barlow_twins_loss(h_p_list[0], h_p_list[2],3)
                loss_3 = self.whitened_barlow_twins_loss(h_p_list[0], h_a,4)
                loss_4 = self.whitened_barlow_twins_loss(h_p_list[1], h_a_1,5)
                loss_5 = self.whitened_barlow_twins_loss(h_p_list[2], h_a_2,6)
                loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2\
                                + self.args.w_loss3 * loss_3 + self.args.w_loss4 * loss_4 + self.args.w_loss5 * loss_5
            elif self.args.fea_ins==True:
                loss_0 = self.ins_loss(h_p_list[0], h_p_list[1])+self.fea_loss(h_p_list[0], h_p_list[1])
                loss_1 = self.ins_loss(h_p_list[0], h_a)+self.fea_loss(h_p_list[0], h_a)
                loss_2 = self.ins_loss(h_p_list[1], h_a_1)+self.fea_loss(h_p_list[1], h_a_1)
                loss_3 = self.ins_loss(h_p_list[0], h_a)+self.fea_loss(h_p_list[0], h_a)
                loss_4 = self.ins_loss(h_p_list[1], h_a_1)+self.fea_loss(h_p_list[1], h_a_1)
                loss_5 = self.ins_loss(h_p_list[2], h_a_2)+self.fea_loss(h_p_list[2], h_a_2)

                loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2 \
                       + self.args.w_loss3 * loss_3 + self.args.w_loss4 * loss_4 + self.args.w_loss5 * loss_5
            elif self.args.fea==True:
                loss_0 = self.fea_loss(h_p_list[0], h_p_list[1])
                loss_1 = self.fea_loss(h_p_list[0], h_a)
                loss_2 = self.fea_loss(h_p_list[1], h_a_1)
                loss_3 = self.fea_loss(h_p_list[0], h_a)
                loss_4 = self.fea_loss(h_p_list[1], h_a_1)
                loss_5 = self.fea_loss(h_p_list[2], h_a_2)

                loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2 \
                       + self.args.w_loss3 * loss_3 + self.args.w_loss4 * loss_4 + self.args.w_loss5 * loss_5
            elif self.args.ins==True:
                loss_0 = self.ins_loss(h_p_list[0], h_p_list[1])
                loss_1 = self.ins_loss(h_p_list[0], h_a)
                loss_2 = self.ins_loss(h_p_list[1], h_a_1)
                loss_3 = self.ins_loss(h_p_list[0], h_a)
                loss_4 = self.ins_loss(h_p_list[1], h_a_1)
                loss_5 = self.ins_loss(h_p_list[2], h_a_2)

                loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2 \
                       + self.args.w_loss3 * loss_3 + self.args.w_loss4 * loss_4 + self.args.w_loss5 * loss_5
            else:
                c = (h_p_list[1]).T @ (h_p_list[0])
                c_0 = (h_p_list[1]).T @ (h_p_list[2])
                c_1 = (h_p_list[0]).T @ (h_p_list[2])
                c_2 = (h_p_list[0]).T @ (h_a)
                c_3 = (h_p_list[1]).T @ (h_a_1)
                c_4 = (h_p_list[1]).T @ (h_a_2)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                loss_0 = on_diag + self.args.lambd0 * off_diag

                on_diag_0 = torch.diagonal(c_0).add_(-1).pow_(2).sum()
                off_diag_0 = off_diagonal(c_0).pow_(2).sum()
                loss_1 = on_diag_0 + self.args.lambd1 * off_diag_0
                #
                on_diag_1 = torch.diagonal(c_1).add_(-1).pow_(2).sum()
                off_diag_1 = off_diagonal(c_1).pow_(2).sum()
                loss_2 = on_diag_1 + self.args.lambd2 * off_diag_1

                on_diag_2 = torch.diagonal(c_2).add_(-1).pow_(2).sum()
                off_diag_2 = off_diagonal(c_2).pow_(2).sum()
                loss_3 = on_diag_2 + self.args.lambd3 * off_diag_2

                on_diag_3 = torch.diagonal(c_3).add_(-1).pow_(2).sum()
                off_diag_3 = off_diagonal(c_3).pow_(2).sum()
                loss_4 = on_diag_3 + self.args.lambd4 * off_diag_3

                on_diag_4 = torch.diagonal(c_4).add_(-1).pow_(2).sum()
                off_diag_4 = off_diagonal(c_4).pow_(2).sum()
                loss_5 = on_diag_4 + self.args.lambd5 * off_diag_4
                loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2\
                                + self.args.w_loss3 * loss_3 + self.args.w_loss4 * loss_4 + self.args.w_loss5 * loss_5
        return loss

    def embed(self, x, adj_list=None,adj_fusion=None):
        if self.args.length == 2:
            h_p = self.MLP1(x)
            h_p_1 = self.MLP2(x)
        elif self.args.length == 3:
            h_p = self.MLP1(x)
            h_p_1 = self.MLP2(x)
            h_p_2 = self.MLP3(x)

        h_p_list = []
        i =0
        for adj in adj_list:
            if self.args.sparse:
                if i == 0:
                    h_p = torch.spmm(adj, h_p)
                if i == 1:
                    h_p = torch.spmm(adj, h_p_1)
                if i == 2:
                    h_p = torch.spmm(adj, h_p_2)
                h_p_list.append(h_p)
            else:
                h_p = torch.mm(adj, h_p)
                h_p_list.append(h_p)
            i += 1
        h_fusion = self.combine_att(h_p_list)

        return  h_fusion.detach()




def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
            # result = nn.Sequential(*layers)
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)#, result


class WhitenTran(nn.Module):
    def __init__(self, eps=0.01, dim=256):
        super(WhitenTran, self).__init__()
        self.eps = eps
        self.dim = dim

    def cholesky_forward(self, x):
        """normalized tensor"""
        batch_size, feature_dim = x.size()
        f_cov = torch.mm(x.transpose(0, 1), x) / (batch_size - 1) # d * d
        eye = torch.eye(feature_dim).float().to(f_cov.device)
        f_cov_shrink = (1 - self.eps) * f_cov + self.eps * eye
        inv_sqrt = torch.triangular_solve(eye, torch.linalg.cholesky(f_cov_shrink.float()), upper=False)[0]
        inv_sqrt = inv_sqrt.contiguous().view(feature_dim, feature_dim).detach()
        return torch.mm(inv_sqrt, x.transpose(0, 1)).transpose(0, 1)    # N * d

    def zca_forward(self, x):
        batch_size, feature_dim = x.size()
        eps = 0.1
        f_cov = (torch.mm(x.transpose(0, 1), x) / (batch_size - 1)).float()  # d * d
        eye = torch.eye(feature_dim).float().to(f_cov.device)
        # f_cov = torch.mm(x.transpose(0, 1), x).float()  # d * d
        U, S, V = torch.linalg.svd(0.9 * f_cov + 0.1 * eye)
        diag = torch.diag(1.0 / torch.sqrt(S+1e-5))
        rotate_mtx = torch.mm(torch.mm(U, diag), U.transpose(0, 1)).detach() # d * d
        return torch.mm(rotate_mtx, x.transpose(0, 1)).transpose(0, 1)  # N * d

    def pca_for(self, x):
        batch_size, feature_dim = x.size()
        f_cov = (torch.mm(x.transpose(0, 1), x) / (batch_size - 1)).float()  # d * d
        eye = torch.eye(feature_dim).float().to(f_cov.device)
        # f_cov = torch.mm(x.transpose(0, 1), x).float()  # d * d
        U, S, V = torch.linalg.svd(0.99 * f_cov + 0.01 * eye)
        diag = torch.diag(1.0 / torch.sqrt(S + 1e-5))
        rotate_mtx = torch.mm(diag, U.transpose(0, 1)).detach()  # d * d
        return torch.mm(rotate_mtx, x.transpose(0, 1)).transpose(0, 1)  # N * d

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cfg = args.cfg
        length = args.length
        self.w_list = nn.ModuleList([nn.Linear(cfg[-1], cfg[-1], bias=True) for _ in range(length)])
        self.y_list = nn.ModuleList([nn.Linear(cfg[-1], 1) for _ in range(length)])
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)


    def combine_att(self, data_list):
        combine = []
        for i,data in enumerate(data_list):
            data=self.w_list[i](data)
            data=self.y_list[i](data)
            combine.append(data)
        score = torch.cat(combine, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        data = torch.stack(data_list, dim=1)
        data = score * data
        data = torch.sum(data, dim=1)
        return data

    def forward(self,data_list):
        data=self.combine_atte(data_list)
        return data