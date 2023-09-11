# -*- coding: utf-8 -*-

import math
import numpy as np
# #from collections import defaultdict
# from utils.mathfunctions import sigmoid
from  sklearn.metrics import roc_auc_score 

class BPR(object):
    '''隐式行为算法'''
    def __init__(self, config, dl):
        self.config = config
        self.dl = dl
        self.lRate = self.config.lr
        self.regU = self.config.regU
        self.regI = self.config.regI1
        self.initModel()
    def initModel(self):
        self.P = np.random.rand(self.dl.num_users, self.config.embedding_size)/3
        self.Q = np.random.rand(self.dl.num_items, self.config.embedding_size)/3
    def buildModel(self):
        for u, uhist in enumerate(self.dl.trainset()):
            history,_ = zip(*uhist)
            for i in history:
                j = np.random.choice(self.dl.num_items)
                while j in history:
                    j =  np.random.choice(self.dl.num_items)
                self._optimize(u,i,j)
    def sigmoid(self, x):
        return 1/(1+math.exp(-x))
       
    def _optimize(self,u,i,j):
        s = self.sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * (1 - s) * self.P[u]
        self.Q[j] -= self.lRate * (1 - s) * self.P[u]

        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]
        self.loss += -math.log(s)
    
    def train_and_evaluate(self):
        print('training...')
        iter = 0
        while iter < self.config.epoches:
            self.loss = 0
            self.buildModel()
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            print('loss:%.10f, P:%.4f'%(self.loss, np.mean(self.P)))
            if iter % 1 ==0 :
                self.evaluate()
            iter += 1 
#            if self.isConverged(iter):
#                break
    def predict(self, u, i):
        yui = self.sigmoid(self.Q[i].dot(self.P[u]))
        return yui
 
         