import pandas as pd
import numpy as np
import re
import pandas as pd
import numpy as np
from sklearn import model_selection
import scipy.sparse as sp
import xgboost as xgb
#[3000]  valid_0's rmse: 0.53221
#[30000] valid_0's rmse: 0.0558356
# 在这里面提取每一个的氨基酸特征

file_name = 'all_info.csv'
df_affinity_train = pd.read_csv('df_affinity_train.csv', encoding='utf-8')
df_affinity_test = pd.read_csv('df_affinity_test_toBePredicted.csv', encoding='utf-8')
df_protein_train = pd.read_csv('df_protein_train.csv', encoding='utf-8')
df_protein_test = pd.read_csv('df_protein_test.csv', encoding='utf-8')
df_molecule = pd.read_csv('df_molecule.csv', encoding='utf-8')
data = pd.concat([df_affinity_train, df_affinity_test], axis=0)
protein_concat = pd.concat([df_protein_train, df_protein_test], axis=0)

target_info = pd.merge(data, protein_concat, on='Protein_ID', how='left')
target_info = pd.merge(target_info, df_molecule, on='Molecule_ID', how='left')
target_info.to_csv(file_name, encoding='utf-8', index=False)



#特征工程
#特征一:氨基酸特征
feature_1 = pd.DataFrame()
target_protein = target_info['Sequence']
all_anjisuan = pd.read_csv('anjisuan.csv', encoding='utf-8')
all_anjisuan = all_anjisuan['anjisuan_first']
#个数count
for i in all_anjisuan:
    temp_list = []
    for j in target_protein:
        temp_list.append(j.count(i))
    feature_1['length_anjisuan' + i] = temp_list

#个数频率
for i in all_anjisuan:
    temp_list = []
    for j in target_protein:
        temp_list.append(float(j.count(i))/len(j))
    feature_1['freq_anjisuan' + i] = temp_list

all_anjisuan = pd.read_csv('anjisuan.csv', encoding='utf-8')
all_name = all_anjisuan['anjisuan_first']
all_sum = all_anjisuan['number_it']
dict_anjisuan = dict(zip(all_name,all_sum))
anjisuan_first = []
for i in target_protein:
    anjisuan_first.append(dict_anjisuan[i.upper()[0]])#开头序列
feature_1['anjisuan_first'] = anjisuan_first

#开头第一第二序列
anjisuan_first = []
for i in target_protein:
    anjisuan_first.append(int(str(dict_anjisuan[i.upper()[0]]) + str(dict_anjisuan[i.upper()[1]])))
feature_1['anjisuan_first_two'] = anjisuan_first

#开头第一第二第三序列
anjisuan_first = []
for i in target_protein:
    anjisuan_first.append(int(str(dict_anjisuan[i.upper()[0]]) + str(dict_anjisuan[i.upper()[1]]) + str(dict_anjisuan[i.upper()[2]])))
feature_1['anjisuan_first_three'] = anjisuan_first
feature_1.to_csv('feature_1.csv', index=False)

#特征二
#序列长度
feature_2=target_info.drop(['Ki','Protein_ID','Sequence','Fingerprint'],axis=1)
target_protein = target_info['Sequence']
target_protein_list = []
for i in target_protein:
    target_protein_list.append(len(i))
feature_2['length_protein'] = target_protein_list

target_Fingerprint = target_info['Fingerprint']

target_fingerprint_list = []
for i in target_Fingerprint:
    target_fingerprint_list.append(i.strip().split(','))

temp = pd.DataFrame(target_fingerprint_list)
for i in np.arange(0, len(target_fingerprint_list[0]), 1):
    feature_2['figerprint' + str(i)] = temp[i]

#计算序列中非0个数
fingerprint_num0 = []
for i in target_fingerprint_list:
    temp = map(int, i)
    fingerprint_num0.append(np.count_nonzero(temp))
feature_2['figerprint_num0'] = fingerprint_num0

#fingerprint_num1 = []
#for i in target_fingerprint_list:
#    temp = map(int, i)
#    fingerprint_num1.append(len(temp) - np.count_nonzero(temp))
#feature_2['figerprint_num1'] = fingerprint_num1


feature_2.to_csv('feature_2.csv', index=False)

#特征三
feature_3 = pd.DataFrame()

#分子指纹前两个
target_Fingerprint = target_info['Fingerprint']
target_fingerprint_list = []
for i in target_Fingerprint:
    t_str = i.replace(' ','').split(',')
    target_fingerprint_list.append(t_str[0] + t_str[1])
feature_3['molecule_first_two'] = target_fingerprint_list

#分子指纹前三个
target_Fingerprint = target_info['Fingerprint']
target_fingerprint_list = []
for i in target_Fingerprint:
    t_str = i.replace(' ','').split(',')
    target_fingerprint_list.append(t_str[0] + t_str[1] + t_str[2])
feature_3['molecule_first_there'] = target_fingerprint_list
feature_3.to_csv('feature_3.csv', index=False)

#特征四
feature_4 = pd.DataFrame()
target_protein = target_info['Sequence']
all_anjisuan = pd.read_csv('anjisuan.csv', encoding='utf-8')
all_name = all_anjisuan['anjisuan_first']
list_name = []
i = 0
while i < len(all_name):
    j = i
    while j < len(all_name):
        if (i == j):
            list_name.append(all_name[i] + all_name[j])
        else:
            list_name.append(all_name[i] + all_name[j])
            list_name.append(all_name[j] + all_name[i])
        j = j + 1
    i = i + 1

for i in list_name:
    temp_list = []
    for j in target_protein:
        temp_list.append(len(re.findall(r'(?=%s)'%(i), j)))
    feature_4[i + 'count'] = temp_list
#del feature_4['LLcount']
#del feature_4['SVcount']
#del feature_4['LAcount']
#del feature_4['SLcount']
feature_4.to_csv('feature_4.csv', index=False)

####################################lgb##########################################
import pandas as pd
import numpy as np
from sklearn import model_selection
import scipy.sparse as sp


df_affinity_train = pd.read_csv('df_affinity_train.csv', encoding='utf-8')
df_affinity_test = pd.read_csv('df_affinity_test_toBePredicted.csv', encoding='utf-8')
feature_1 = pd.read_csv('feature_1.csv', encoding='utf-8')
feature_2 = pd.read_csv('feature_2.csv', encoding='utf-8')
feature_3 = pd.read_csv('feature_3.csv', encoding='utf-8')
feature_4 = pd.read_csv('feature_4.csv', encoding='utf-8')

feature = pd.concat([feature_1, feature_2,feature_3, feature_4], axis=1)

#feature [206467 rows x 629 columns]
train_feature = feature[:len(df_affinity_train)]
test_feature = feature[len(df_affinity_train):]
label = list(df_affinity_train['Ki'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, label, test_size=0.2,random_state=1017)

############################xgb#######################
#train_feat = train_feat.drop('index',axis=1)
feature1_xy=xgb.DMatrix(train_feature, label)

#s=list(train_feat.columns)
#ss=list(set(train_feat.columns))

params={'booster':'gbtree',
	    'objective': 'reg:linear',
	    'eval_metric':'rmse',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }
     
watchlist = [(feature1_xy, 'train')]
model = xgb.train(params, feature1_xy, num_boost_round=3000,evals=watchlist)

#save feature score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('nurbs_xgb_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)
#######################################

####################### lgb ##########################
#import lightgbm as lgb
#
#
#
#lgb_train = lgb.Dataset(train_feature, label)
#lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
#
#params = {
#    'task': 'train',
#    'boosting_type': 'gbdt',
#    'objective': 'regression',
#    'metric': {'rmse'}
#}
#gbm = lgb.train(params,
#                lgb_train,
#                num_boost_round=3000,
#                valid_sets=lgb_eval
#                )
#
##save feature score
#feature_score = gbm.get_fscore()
#feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
#fs = []
#for (key,value) in feature_score:
#    fs.append("{0},{1}\n".format(key,value))
#    
#with open('nurbs_xgb_feature_score.csv','w') as f:
#    f.writelines("feature,score\n")
#    f.writelines(fs)
#
#    
##preds_prob = gbm.predict(test_feature)
##
##feature_name = train_feature.columns
##feature_imp = gbm.feature_importance()
##feature_name = feature_name[np.argsort(feature_imp)]
##feature_imp = feature_imp[np.argsort(feature_imp)]
##
##df_result = pd.DataFrame()
##df_result['Protein_ID'] = df_affinity_test['Protein_ID']
##df_result['Molecule_ID'] = df_affinity_test['Molecule_ID']
##df_result['Ki'] = preds_prob
##df_result.to_csv('lgb_result.csv', header=True, index=False)
#
#
