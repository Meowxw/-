#encoding=utf-8
import pandas as pd
import numpy as np
import sys

#1.蛋白质信息 df_protein.csv
#2.蛋白质小分子亲和力值信息 df_affinity.csv 
#3.小分子信息 df_molecule.csv（会存在缺失值） 

#len(set(df_affinity_train['Molecule_ID']))
#Out[58]: 98299

#len(set(df_affinity_test['Molecule_ID']))
#Out[59]: 34839

df_protein_train = pd.read_csv('df_protein_train.csv')
df_protein_train['seq_len'] = df_protein_train['Sequence'].apply(len)

df_protein_test = pd.read_csv('df_protein_test.csv')
df_protein_test['seq_len'] = df_protein_test['Sequence'].apply(len)

df_molecule = pd.read_csv('df_molecule.csv')
df_molecule['Fingerprint_len'] = df_molecule['Fingerprint'].apply(len)

df_affinity_train = pd.read_csv('df_affinity_train.csv')

df_affinity_test = pd.read_csv('df_affinity_test_toBePredicted.csv')

df_molecule_avg = df_affinity_train.groupby(['Molecule_ID'],as_index=False)['Ki'].agg({'Ki_avg':'mean'})

df1 = pd.merge(df_affinity_train,df_protein_train, on=["Protein_ID"],how='left')

df1 = pd.merge(df_affinity_train,df_protein_test, on=["Protein_ID"],how='left')

df1 = pd.merge(df_affinity_test,df_molecule_avg, on=["Molecule_ID"],how='left')

df1 = df1.fillna(df1.mean()['Ki_avg'])

df1.columns = ['Protein_ID','Molecule_ID','Ki']

df1.to_csv("result.csv",index=False)