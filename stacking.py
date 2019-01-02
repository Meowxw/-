import gc
import re
import sys
import time
import jieba
import os.path
import os
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import gensim 
from gensim.models import Word2Vec  
#0.65 0.35:1.26873
#0.75 0.25:1.27***1
#0.55 0.45:1.27***8
#0.60 0.40:1.27
#0.63 0.37:1.26***5
#0.64 0.36:1.26***4
#0.62 0.38:1.26***4
#0.67 0.33:1.26***9
#0.66 0.34:1.26***2

#1.27***6 0.6,0.25.0.1,0.05
#1.28***7 0.65,0.2.0.15
#1.31***9 0.65,0.2,0,0.15
x1=pd.read_csv('lgb_result.csv')
x2=pd.read_csv('lgb_0001.csv')
x3=pd.read_csv('result.csv')
#x3=pd.read_csv('lda_1.5.csv')
#x4=pd.read_csv('cnn_2.xx.csv')
x=pd.merge(x1,x2,on=['Protein_ID','Molecule_ID'],how='left')
x.rename(columns={'Ki_x':'Ki_1','Ki_y':'Ki_2'},inplace=True)
x=pd.merge(x,x3,on=['Protein_ID','Molecule_ID'],how='left')
x.rename(columns={'Ki':'Ki_3'},inplace=True)
#x=pd.merge(x,x4,on=['Protein_ID','Molecule_ID'],how='left')
#x.rename(columns={'Ki':'Ki_4'},inplace=True)

x['Ki']=x['Ki_1']*0.35+x['Ki_2']*0.35+x['Ki_3']*0.3
x=x[['Protein_ID','Molecule_ID','Ki']]
x.to_csv('x.csv',index=None)