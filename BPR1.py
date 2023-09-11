import numpy
import torch
import os
import random
from collections import defaultdict
from sklearn.metrics import auc,roc_auc_score
def load_data(data_path):
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open(data_path, 'r') as f:
        for line in f.readlines():
            u, i, _, _ = line.split("\t")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
    print ("max_u_id:", max_u_id)
    print ("max_i_id:", max_i_id)
    return max_u_id, max_i_id, user_ratings
    
data_path ='./ml-100k/u.data' 
# os.path.join('./ml-100k', 'u.data')
user_count, item_count, user_ratings = load_data(data_path)

def generate_test(user_ratings):
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test

user_ratings_test = generate_test(user_ratings)
def generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512):
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]
        j = random.randint(1, item_count)
        while j in user_ratings[u]:
            j = random.randint(1, item_count)
        t.append([u, i, j])
    return numpy.asarray(t)

def generate_test_batch(user_ratings, user_ratings_test, item_count):
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(1, item_count+1):
            if not (j in user_ratings[u]):
                t.append([u, i, j])
        yield numpy.asarray(t)

hidden_dim = 20
user_emb_w = torch.nn.Parameter(torch.rand((user_count+1,hidden_dim),dtype=torch.float32))
item_emb_w = torch.nn.Parameter(torch.rand((item_count+1,hidden_dim),dtype=torch.float32))
optimizer = torch.optim.SGD([user_emb_w,item_emb_w], lr=0.01, momentum=0.9)

# training 
for epoch_idx in range(4):
    loss_mean = 0
    for batch_idx in range(5000):
        uij = generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512)
        u,i,j = torch.tensor(uij[:,0],dtype=torch.int64),torch.tensor(uij[:,1],dtype=torch.int64),torch.tensor(uij[:,2],dtype=torch.int64)
        u_emb = user_emb_w[u]
        i_emb = item_emb_w[i]
        j_emb = item_emb_w[j]
        optimizer.zero_grad()
        bprloss= -torch.mean(torch.log(torch.sigmoid(torch.sum(u_emb*(i_emb-j_emb),dim = 1))))
        bprloss.backward()
        optimizer.step()
        loss_mean+=bprloss.data
    
    preds = []
    for uij_test in generate_test_batch(user_ratings, user_ratings_test, item_count):
        u,i,j = torch.tensor(uij_test[:,0],dtype=torch.int64),torch.tensor(uij_test[:,1],dtype=torch.int64),torch.tensor(uij_test[:,2],dtype=torch.int64)
        u_emb = user_emb_w[u]
        i_emb = item_emb_w[i]
        j_emb = item_emb_w[j]
        pred = torch.sigmoid(torch.sum(u_emb*(i_emb-j_emb),dim = 1))
        preds.extend(pred.tolist())
    preds_num = len(preds)
    print("Epoch {}, loss:{}, acc:{}".format(epoch_idx,loss_mean*1.0/5000,len([v for v in preds if v>=0.5])/preds_num))

### output
# Epoch 0, loss:0.536064624786377, acc:0.8043799785482523
# Epoch 1, loss:0.38312220573425293, acc:0.836085903887019
# Epoch 2, loss:0.34411370754241943, acc:0.8450205433455844
# Epoch 3, loss:0.3259732723236084, acc:0.8485525453427233

# test
x= torch.matmul(user_emb_w,item_emb_w.T)
# print(x.shape,x[0])
value,indices = torch.topk(x[0],5)
for i in range(len(value.data)):
    print("the item id: {}, the score: {}".format(indices.data[i],value.data[i]))