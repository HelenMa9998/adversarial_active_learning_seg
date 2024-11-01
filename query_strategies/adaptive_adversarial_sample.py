import math
import numpy as np
import torch
from .strategy import Strategy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
# This is our proposed method
# including pseudo labeling expansion, generated sample expansion and adaptive selection

from distutils.util import change_root
import numpy as np
import torch
from .strategy import Strategy
from tqdm import tqdm

# DeepFool tries to assume that the neural network is completely linear, considering that the neural network divides the space where the training data is located into different regions through the hyperplane,
# Each region belongs to a class. Based on this assumption, the core idea of DeepFool is to find the minimum adversarial perturbation that can make the sample cross the classification boundary by iterating continuously
# The minimum perturbation to change the classification of a sample x is to move x to the hyperplane, and the distance from this sample to the hyperplane is the least costly place.

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# criterion = nn.BCELoss()

def count(nx):
    zeros=0
    ones=0
    for i in nx:
        if i==0:
            zeros += 1
        elif i==1:
            ones += 1
    return zeros,ones

class AdversarialAttack(Strategy):
    def __init__(self, dataset, net, eps=0.05, max_iter=1):
        super(AdversarialAttack, self).__init__(dataset, net)
        self.eps = eps
        self.max_iter = max_iter

    def cal_dis(self, x, unlabeled_idxs,criterion):
        nx = torch.unsqueeze(x, 0)
        nx = torch.unsqueeze(nx, 0)
        nx = nx.cuda()
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)
        eta = eta.cuda()
        out,_ = self.net.clf(nx+eta)#seg 3d batchsize,w,h,2
        out_copy = out.clone()
        out_binary = (out_copy > 0.5).int() 
        py = out_binary#segmentation predicted by network batchsize,w,h,d ([1, 1, 128, 128])
        ny = out_binary#([1, 1, 128, 128])
        i_iter = 0
        change_pixel_num = 0

        while i_iter < self.max_iter:#设置最多次iter

            loss = criterion(out.float(), ny.float())
            
            loss.backward(retain_graph=True)
            eta += self.eps * nx.grad.data/nx.grad.data.max()#用了gradient的大小+符号
            nx.grad.data.zero_()
            out_change,_ = self.net.clf(nx + eta)

            out_copy_change = out_change.clone()
            change_pixel = torch.ne(out_copy_change.flatten(),out_copy.flatten())
            change_pixel_num = change_pixel.tolist().count(True)
            i_iter += 1
        # image = (nx + eta).cpu().detach()  # 最错误的样本

        return (eta*eta).sum(), change_pixel_num

    def query(self, n, index, param2,param3):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(index = index)
        print("unlabeled_idxs",len(unlabeled_idxs))
        criterion = FocalLoss(alpha=param3, gamma=2,logits=False)
        self.clf = torch.load('./result/model.pth')
        self.net.clf.eval()
        dis = np.zeros(unlabeled_idxs.shape)
        changed = np.zeros(unlabeled_idxs.shape)
        # generated_image = {}
        # image = torch.zeros((len(unlabeled_idxs),1,128,128))#([1, 1, 128, 128])
        # final_image = torch.zeros((500,1,128,128))
        diction = {}
        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):#从1开始
            x, y, idx = unlabeled_data[i]
            dis[i],changed[i] = self.cal_dis(x,i,criterion)
            # generated_image[unlabeled_idxs[i]] = image[i]

        idx_dis = np.array(unlabeled_idxs[dis.argsort()])#distance由小到大 返回index[12885 17111 17112 ...  8612  8477     1]

        idx_changed = np.array(unlabeled_idxs[changed.sort()][0])#changed由小到大 返回index[    1     2     4 ... 25534 25535 25536]
        for i in range(len(idx_dis)):#两个index中是真正的idx顺序
            position = int(np.where(idx_changed==idx_dis[i])[0])+i#pixel+noise的排位
            diction[position] = idx_dis[i]#存储position对其原本index；position是key，idx是value

        dict_sort = sorted(diction.keys())#对pixel+noise的排位sort
        keys = dict_sort[:n]
        final = [value for key, value in diction.items() if key in keys]
        # for i in range(len(final)):
        #     final_image[i] = generated_image[final[i]]
        return final