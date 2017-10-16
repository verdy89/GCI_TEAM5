# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 18:58:36 2017

@author: MEIP-users
"""

import pandas as pd
import os
import numpy as np
import numpy.matlib
from operator import itemgetter
from scipy import sparse
import sklearn.preprocessing as pp
import csv  
import codecs 

now=os.getcwd()
os.chdir("./train") # ディレクトリに移動

#traindataの読み込み
dfa = pd.read_csv('train_A.tsv', sep='\t')
dfb = pd.read_csv('train_B.tsv', sep='\t')
dfc = pd.read_csv('train_C.tsv', sep='\t')
dfd = pd.read_csv('train_D.tsv', sep='\t')

#行userid,列productidの行列、購買行動の点数を合計(要検討)
#A,aにはそれぞれ'_A','_a'が入る
def makematrix(df,A,a):
    shape = (int(df.max().ix['user_id'].rstrip(A)) + 1, int(df.max().ix['product_id'].rstrip(a)) + 1)
    R = sparse.lil_matrix(shape) 
    for i in df.index:
        row = df.ix[i]
        R[int(row['user_id'].rstrip(A)), int(row['product_id'].rstrip(a))] += row['event_type']
    return R

Ra=makematrix(dfa,'_A','_a')
Rb=makematrix(dfb,'_B','_b')
Rc=makematrix(dfc,'_C','_c')
Rd=makematrix(dfd,'_D','_d')

#計算高速化
Ra1=Ra.tocsc()
Rb1=Rb.tocsc()
Rc1=Rc.tocsc()
Rd1=Rd.tocsc()

#協調フィルタリングの類似度行列sims
def cosine_similarities_user(mat):
    col_normed_mat = pp.normalize(mat, axis=1)
    return col_normed_mat * col_normed_mat.T
    
simsa2=cosine_similarities_user(Ra1)
simsb2=cosine_similarities_user(Rb1)
simsc2=cosine_similarities_user(Rc1)
simsd2=cosine_similarities_user(Rd1)
   
#類似度の高い上位kユーザを抽出(冷静にsimsにRかければいい気もする)
#uはuser_id
def usertorec(u,sims2,a,R1):
    k=20
    #sparseからdenseにして並べ替える
    a2=np.array(sims2[u,:].todense())[0]
    b2 = np.sort(a2)[::-1]
    x2 = np.argsort(a2)[::-1]
    #uに対して推薦するアイテムを選ぶ(上位kユーザーが買ったアイテムを類似度を掛けて合計で考える(要検討))
    itemmat = sparse.lil_matrix((1, R1.shape[1]))
    for i in x2[0:k]:
        itemmat+=R1[i,:]*a2[i]
    #sparseからdenseにして並べ替える
    itemrec=np.array(itemmat.todense())[0]
    itemrecr = np.sort(itemrec)[::-1]
    itemrec1 = np.argsort(itemrec)[::-1]
    return itemrec1[0:20]
    
rows=[]
#ひとつディレクトリ下げる
os.chdir(now)
testid = pd.read_csv('test.tsv', sep='\t')
recnum=20
for ui in testid['user_id']:
    if 'A' in ui:
        u=int(ui.rstrip('_A'))
        recsa=usertorec(u,simsa2,'_a',Ra1)
        for i in range(recnum):
            rows.append((ui, "{0:08d}".format(recsa[i])+'_a',str(i)))
    elif 'B' in ui:
        u=int(ui.rstrip('_B'))
        recsb=usertorec(u,simsb2,'_b',Rb1)
        for i in range(recnum):
            rows.append((ui, "{0:08d}".format(recsb[i])+'_b',str(i)))
    elif 'C' in ui:
        u=int(ui.rstrip('_C'))
        recsc=usertorec(u,simsc2,'_c',Rc1)
        for i in range(recnum):
            rows.append((ui, "{0:08d}".format(recsc[i])+'_c',str(i)))
    elif 'D' in ui:
        u=int(ui.rstrip('_D'))
        recsd=usertorec(u,simsd2,'_d',Rd1)
        for i in range(recnum):
            rows.append((ui, "{0:08d}".format(recsd[i])+'_d',str(i)))

#カキコ
class CustomFormat(csv.excel):  
    #quoting   = csv.QUOTE_ALL  
    delimiter = '\t' 
csv_file = codecs.open('./test_rec.tsv', 'w', 'shift_jis') 
writer = csv.writer(csv_file, CustomFormat()) 
writer.writerows(rows)
csv_file.close()