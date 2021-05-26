import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
from torch.nn import functional as F
from utils.helpers import initialize_weights, set_trainable

def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu
    
def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)

def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs-1):
        result = torch.cat((result, distance(a[i], b)), 0)
        
    return result

def multiply(x): #to flatten matrix into a vector 
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)

def index(batch_size, x):
    idx = torch.arange(0, batch_size).long() 
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)

def MemoryLoss(memory):

    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t))/2 + 1/2 # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)
    
    return torch.sum(sim)/(m*(m-1))


class Memory_unsup(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim,  temp_update, temp_gather,supervised_mem,momentum):
        super(Memory_unsup, self).__init__()
        # Constants
        self.memory_size = memory_size # when supervised memory, set same number with class num(19)
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        self.supervised_mem = supervised_mem
        self.momentum = momentum
        self.output = nn.Sequential( # refer object contextual represenation network fusion layer...
                nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        initialize_weights(self)
        
    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem,torch.t(self.keys_var))
        similarity[:,i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)
        
        
        return self.keys_var[max_idx]
    
    def random_pick_memory(self, mem, max_indices):
        
        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices==i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)
            
        return torch.tensor(output)
    
    def get_update_query(self, mem, max_indices, update_indices, score, query):

        m, d = mem.size()
        query_update = torch.zeros((m,d)).cuda()
        random_update = torch.zeros((m,d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1)==i)
            a, _ = idx.size()
            #ex = update_indices[0][i]
            if a != 0:
                #random_idx = torch.randperm(a)[0]
                #idx = idx[idx != ex]
#                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) *query[idx].squeeze(1)), dim=0)
                #random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
            else:
                query_update[i] = 0
                #random_update[i] = 0
        
       
            return query_update 


    def get_score(self, mem, query):
        bs, h,w,d = query.size()
        m, d = mem.size()
        
        score = torch.matmul(query, torch.t(mem))# b X h X w X m
        score = score.view(bs*h*w, m)# (b X h X w) X m
        
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score,dim=1)
        
        return score_query, score_memory
    
    def forward(self, query, keys): # doesn't update memory in forward

        batch_size, dims,h,w = query.size() # b X d X h X w
        query = F.normalize(query, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d

        #gathering loss
        gathering_loss = self.gather_loss(query,keys)
        #spreading_loss
        spreading_loss = self.spread_loss(query, keys)
        # read
        updated_query, softmax_score_query,softmax_score_memory = self.read(query, keys)

        return updated_query, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss


    
    def update(self, query, keys,mask = None):

        batch_size, dims, h, w = query.size()
        query = F.normalize(query, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d

        batch_size, h,w,dims = query.size() # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)

        # top-1 queries (of each memory) update (weighted sum) & random pick
        query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape)
        updated_memory = F.normalize(query_update + keys, dim=1)

        # else:
        #     # only weighted sum update when test
        #     query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape, train)
        #     updated_memory = F.normalize(query_update + keys, dim=1)

        # top-1 update
        #query_update = query_reshape[updating_indices][0]
        #updated_memory = F.normalize(query_update + keys, dim=1)
      
        return updated_memory.detach()

        
    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n,dims = query_reshape.size() # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')
        
        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())
                
        return pointwise_loss
        
    def spread_loss(self,query, keys):
        batch_size, h,w,dims = query.size() # b X h X w X d

        loss = torch.nn.TripletMarginLoss(margin=1.0)

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

        #1st, 2nd closest memories
        pos = keys[gathering_indices[:,0]]
        neg = keys[gathering_indices[:,1]]

        spreading_loss = loss(query_reshape,pos.detach(), neg.detach())

        return spreading_loss
        
    def gather_loss(self, query, keys):
        
        batch_size, h,w,dims = query.size() # b X h X w X d

        loss_mse = torch.nn.MSELoss()

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss
            

    def read(self, query, updated_memory):
        batch_size, h,w,dims = query.size() # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory) # (b X h X w) X d
        updated_query = torch.cat((query_reshape, concat_memory), dim = 1) # (b X h X w) X 2d
        updated_query = updated_query.view(batch_size, h, w, 2*dims)
        updated_query = updated_query.permute(0,3,1,2)

        updated_query = self.output(updated_query)
        
        return updated_query, softmax_score_query, softmax_score_memory


class Memory_sup(nn.Module):
    def __init__(self, memory_size, feature_dim, momentum,temperature, add1by1,clsfy_loss,gumbel_read):
        super(Memory_sup, self).__init__()
        # Constants
        self.memory_size = memory_size  # when supervised memory, set same number with class num(19)
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.temperature = temperature
        self.output = nn.Sequential(  # refer object contextual represenation network fusion layer...
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        if add1by1:
            self.writefeat = nn.Sequential(  # refer object contextual represenation network fusion layer...
                nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        else:
            self.writefeat = lambda x: x.clone()

        if clsfy_loss:
            self.mem_cls = torch.tensor([x for x in range(self.memory_size)]).cuda()
            self.clsfier = nn.Linear(in_features=self.feature_dim, out_features=self.memory_size, bias=True)
        else:
            self.clsfier = None

        self.celoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.gumbel_read = gumbel_read
        initialize_weights(self)

    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem, torch.t(self.keys_var))
        similarity[:, i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)

        return self.keys_var[max_idx]

    def random_pick_memory(self, mem, max_indices):

        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices == i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)

        return torch.tensor(output)

    def get_update_query(self, mem, max_indices, update_indices, score, query):

        m, d = mem.size()
        query_update = torch.zeros((m, d)).cuda()
        random_update = torch.zeros((m, d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i)
            a, _ = idx.size()
            # ex = update_indices[0][i]
            if a != 0:
                # random_idx = torch.randperm(a)[0]
                # idx = idx[idx != ex]
                #                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)), dim=0)
                # random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
            else:
                query_update[i] = 0
                # random_update[i] = 0

            return query_update

    def get_score(self, mem, query):
        bs, h, w, d = query.size()
        m, d = mem.size()

        score = torch.matmul(query, torch.t(mem))  # b X h X w X m
        score = score.view(bs * h * w, m)  # (b X h X w) X m

        score_query = F.softmax(score, dim=0) # 특정 메모리 슬롯이 들어오는 query들과 어떤 관계인지.
        score_memory = F.softmax(score, dim=1) # 특정 query vector과 메모리들과 어떤 관계인지.

        return score_query, score_memory


    def get_gumbel_score(self, mem, query):
        bs, h, w, d = query.size()
        m, d = mem.size()

        score = torch.matmul(query, torch.t(mem))  # b X h X w X m
        score = score.view(bs * h * w, m)  # (b X h X w) X m

        # using Gumbel softmax for categorical memory sampling.
        score_query = F.gumbel_softmax(score, dim=0) # 특정 메모리 슬롯이 들어오는 query들과 어떤 관계인지. 확률을 sampling.
        score_memory = F.gumbel_softmax(score, dim=1) # 특정 query vector과 메모리들과 어떤 관계인지.

        return score_query, score_memory


    def get_reading_loss(self,mem,query,mask):
        ### get reading loss between Mt & reading feature
        tempmask = self.make_feature_mask(mask, query.size()[2:])
        query = F.normalize(query, dim=1)
        # query and mem is L2 normalized feature
        query = query.permute(0, 2, 3, 1)
        score = torch.matmul(query, torch.t(mem))/self.temperature  # b X h X w X m
        score = score.permute(0, 3, 1, 2)  # b X m X h X w
        celoss = self.celoss(score, tempmask[:, 0])
        return celoss

    def forward(self, query, key, reading_detach):  # doesn't update memory in forward
        batch_size, dims, h, w = query.size()  # b X d X h X w
        query = F.normalize(query, dim=1)
        # read
        query = query.permute(0, 2, 3, 1)  # b X h X w X d
        updated_query, softmax_score_query, softmax_score_memory = self.read(query, key,reading_detach)

        return updated_query, softmax_score_query, softmax_score_memory



    def make_feature_mask(self,mask,size):
        # making feature scale detached segmentation mask.
        tempmask = F.interpolate(mask.type(torch.float32), size, mode='bilinear', align_corners=True).type(
            torch.int64)
        return tempmask.clone().detach()

    def update(self, query, keys, mask, writing_detach):
        keys = keys.detach() # writing은 항상 전의 메모리 gradient를 끊는다.

        batch_size, dims, h, w = query.size()
        tempmask = self.make_feature_mask(mask,query.size()[2:])

        ### get writing feature.
        query = self.writefeat(query)
        query = F.normalize(query, dim=1)

        ### update supervised memory
        query = query.view(batch_size, dims, -1)
        tempmask[tempmask == -1] = self.memory_size  # when supervised memory, memory size = class number
        tempmask = F.one_hot(tempmask, num_classes=self.memory_size + 1).squeeze()
        tempmask = tempmask.view(batch_size, -1, self.memory_size + 1).type(torch.float32)
        denominator = tempmask.sum(1).unsqueeze(dim=1)
        nominator = torch.matmul(query, tempmask)

        nominator = torch.t(nominator.sum(0)) # batchwise sum
        denominator = denominator.sum(0) # batchwise sum
        denominator = denominator.squeeze()

        updated_memory = keys.clone().detach()
        for slot in range(self.memory_size):
            if denominator[slot] != 0:
                updated_memory[slot] = self.momentum * keys[slot] + ((1 - self.momentum) * nominator[slot]/denominator[slot]) # memory momentum update

        updated_memory = F.normalize(updated_memory,dim=1) # normalize.

        ### get diversity loss about updated memory.
        div_loss = self.diversityloss(updated_memory)

        ### get classification loss about updated memory
        if type(self.clsfier) == type(None):
            cls_loss = torch.tensor(0).cuda()
        else:
            cls_loss = self.classification_loss(updated_memory)

        writing_loss =[div_loss,cls_loss]

        if writing_detach:
            return updated_memory.detach(), writing_loss
        else:
            return updated_memory, writing_loss

    def classification_loss(self,mem):

        score = self.clsfier(mem)
        return self.celoss(score,self.mem_cls)


    def diversityloss(self,mem):

        cos_sim = torch.matmul(mem, torch.t(mem))  # m x m
        loss = (torch.sum(cos_sim) - torch.trace(cos_sim))/(self.memory_size*(self.memory_size-1))
        return loss


    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n, dims = query_reshape.size()  # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')

        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return pointwise_loss

    def spread_loss(self, query, keys):
        batch_size, h, w, dims = query.size()  # b X h X w X d

        loss = torch.nn.TripletMarginLoss(margin=1.0)

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

        # 1st, 2nd closest memories
        pos = keys[gathering_indices[:, 0]]
        neg = keys[gathering_indices[:, 1]]

        spreading_loss = loss(query_reshape, pos.detach(), neg.detach())

        return spreading_loss

    def gather_loss(self, query, keys):

        batch_size, h, w, dims = query.size()  # b X h X w X d

        loss_mse = torch.nn.MSELoss()

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss

    def read(self, query, mem,reading_detach):
        batch_size, h, w, dims = query.size()  # b X h X w X d

        if self.gumbel_read:
            softmax_score_query, softmax_score_memory = self.get_gumbel_score(mem, query)
        else:
            softmax_score_query, softmax_score_memory = self.get_score(mem, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        if reading_detach:
            concat_memory = torch.matmul(softmax_score_memory.detach(), mem)  # (b X h X w) X d
        else:
            concat_memory = torch.matmul(softmax_score_memory, mem)  # (b X h X w) X d
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)  # (b X h X w) X 2d
        updated_query = updated_query.view(batch_size, h, w, 2 * dims)
        updated_query = updated_query.permute(0, 3, 1, 2)

        updated_query = self.output(updated_query)

        return updated_query, softmax_score_query, softmax_score_memory