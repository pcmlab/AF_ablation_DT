#FOAA

import torch
import torch.nn as nn
from torchvision import models
import math as m
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#from Cross_Attention import cross_att #get cross attention file 
#%%
def outer_add(Q, K):
    Q_K_matmul = Q.view(Q.shape[0], Q.shape[1], -1)  + K.view(K.shape[0], K.shape[2],K.shape[1])
    return Q_K_matmul

def outer_sub(Q, K):
    Q_K_matmul = Q.view(Q.shape[0], Q.shape[1], -1)  - K.view(K.shape[0], K.shape[2],K.shape[1])
    return Q_K_matmul

def outer_pro(Q, K):
    Q_K_matmul = Q.view(Q.shape[0], Q.shape[1], -1)  * K.view(K.shape[0], K.shape[2],K.shape[1])
    return Q_K_matmul

def mask_div(x1):
    mask = ((x1  > 0).float() - 1) * 9999  # for -inf
    result = (x1 + mask).softmax(dim=-1)
    return result   

def outer_div(Q, K):
    K = mask_div(K)
    Q_K_matmul = Q.view(Q.shape[0], Q.shape[1], -1)  / K.view(K.shape[0], K.shape[2],K.shape[1])
    return Q_K_matmul

class cross_att(nn.Module):
    # default dim of the model is 8 and head is 4
    def __init__(self, attention_type, d_model=8, num_heads=4, dropout=0.01):
        super().__init__()
        self.d = d_model//num_heads
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.dropout = nn.Dropout(dropout)
        ##create a list of layers for K, and a list of layers for V for each head
        self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.linear_Vs = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.mha_linear = nn.Linear(d_model, d_model)

    def attention(self, Q, K, V,):
        
        if self.attention_type == 'OA': 
            scores = outer_add(Q, K) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        elif self.attention_type == 'OP':
            scores = outer_pro(Q, K) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        elif self.attention_type == 'OD':
            scores = outer_div(Q, K) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        elif self.attention_type == 'OS':
            scores = outer_sub(Q, K) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        else:
            print('attention type has not been selected')
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, x,x2):
        

        Q = [linear_Q(x2) for linear_Q in self.linear_Qs] # Query is from modality x2 
        K = [linear_K(x) for linear_K in self.linear_Ks] # K & V are from x 
        V = [linear_V(x) for linear_V in self.linear_Vs]
        output_per_head = []
        attn_weights_per_head = []

        for Q_, K_, V_ in zip(Q, K, V):
            output, attn_weight = self.attention(Q_, K_, V_)
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)


        output = torch.cat(output_per_head, -1)
        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)
        projection = self.dropout(self.mha_linear(output))
        return projection




#%%
class FOAA(nn.Module):
    def __init__(self,atttention_OA,atttention_OP,atttention_OS,atttention_OD,x1,x3,nb_classes=2, h_dim=1):
        super(FOAA, self).__init__()
        
        self.x1 =  x1
        self.x3 = x3
        self.atttention_OA = atttention_OA
        self.atttention_OP = atttention_OP
        self.atttention_OS = atttention_OS
        self.atttention_OD = atttention_OD
        
        self.att1 = cross_att(atttention_OA,d_model=h_dim, num_heads=h_dim) # this to derive key from la modality for addition operation
        self.att2 = cross_att(atttention_OA,d_model=h_dim, num_heads=h_dim) # this to derive key from ra modality for addition operation
        
        self.att3 = cross_att(atttention_OP,d_model=h_dim, num_heads=h_dim) # this to derive key from la modality for product operation
        self.att4 = cross_att(atttention_OP,d_model=h_dim, num_heads=h_dim) # this to derive key from ra modality for product operation
        
        self.att5 = cross_att(atttention_OS,d_model=h_dim, num_heads=h_dim) # this to derive key from la modality for subtraction operation
        self.att6 = cross_att(atttention_OS,d_model=h_dim, num_heads=h_dim) # this to derive key from ra modality for subtraction operation
        
        self.att7 = cross_att(atttention_OD,d_model=h_dim, num_heads=h_dim) # this to derive key from la modality for subtraction operation
        self.att8 = cross_att(atttention_OD,d_model=h_dim, num_heads=h_dim) # this to derive key from ra modality for subtraction operation
    
    def forward(self, x1,x3):   
            # we need to have our feature map to be of size (batchsize,feature_dim,1) to work with moab,thus we unsqueeze (change dimesnions)
            #Code entering at this point has the dimensions ([80,32]) which become ->([80,32,1]) where format is ([batchsize, features,1])
            x1 = torch.unsqueeze(x1, 2) 
            x3 = torch.unsqueeze(x3, 2)
            
            ### 2) Cross Attention Outer Addition
            x_add_la = self.att1(x1,x3)  
            x_add_ra = self.att2(x3,x1) 
    
            ### 3) Cross Attention Outer Product
            x_prod_la = self.att3(x1,x3) 
            x_prod_ra = self.att4(x3,x1)
        
             ### 4) Cross Attention Outer Subtraction
            x_sub_la = self.att5(x1,x3) 
            x_sub_ra = self.att6(x3,x1)
        
            ### 5) Cross Attention Outer Division
            x_div_la = self.att5(x1,x3) 
            x_div_ra = self.att6(x3,x1)
    
            ### 6) Aggregate FOAA enhanced features 
            x = torch.sum(torch.stack([x_add_la,x_add_ra,x_prod_la,x_prod_ra,x_sub_la,x_sub_ra,x_div_la,x_div_ra,x1,x3]), dim=0) # This operation will perform element-wise addition, maintaining the original size of the vector.
            #x = x.flatten(start_dim=1) #flatten from feature dimension onwards 

            return x  

    
    
