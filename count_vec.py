import re
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition, ensemble
import xgboost
import jieba

#定义路径
# train_path="D:/jupyter_workfile/haizhi/nlp/data/train_data.txt"
# valid_path="D:/jupyter_workfile/haizhi/nlp/data/testset.txt"
# test_path="D:/jupyter_workfile/haizhi/nlp/data/unlabeled_data.txt"
stopwords_path="D:/jupyter_workfile/haizhi/nlp/data/stopwords.txt" #网上载的停用词表
train_path="D:/jupyter_workfile/haizhi/nlp/data/train_modify.txt"
valid_path="D:/jupyter_workfile/haizhi/nlp/data/testset_modify.txt"
test_path="D:/jupyter_workfile/haizhi/nlp/data/unlabeled_modify.txt"
#数据集处理
def data_process(path):
    with open(path,"r",encoding='utf8') as f:
        words = f.read()
    pattern1 = re.compile('(.*?)百度快照__label__positive')
    pattern2 = re.compile('(.*?)百度快照__label__negative')
    po = re.findall(pattern1, words)
    ne = re.findall(pattern2, words)
    print('len-positive:',len(po), 'len-negative:',len(ne))
    x=po+ne
    y=[1]*len(po)+[0]*len(ne)
    x=pd.Series(x)
    y=pd.Series(y)
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(y)
    return x,y,words

train_x,train_y,words_tr=data_process(train_path)
valid_x,valid_y,words_val=data_process(valid_path)

with open(test_path,"r",encoding='utf8') as tv:
    words_test = tv.read()

stopwords = [line.strip() for line in open(stopwords_path, 'r',encoding='utf8').readlines()]

#去停用词，df-itf特征提取
cropus=[]
for i in range(len(train_x)):
    cropus.append(" ".join(jieba.cut(train_x.iloc[i], cut_all=False)))
#for i in range(len(valid_x)):
#    cropus.append(" ".join(jieba.cut(valid_x.iloc[i], cut_all=False)))

#创建一个向量计数器对象
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{2,}',stop_words=stopwords)
count_vect.fit(cropus)
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

test_x=[line.strip() for line in open(test_path,"r",encoding='utf8').readlines()]
test_x=pd.Series(test_x)
xtest_count = count_vect.transform(test_x)

def train_model(classifier, feature_vector_train, train_label, feature_vector_valid,valid_label=None):
    classifier.fit(feature_vector_train, train_label)
    predictions = classifier.predict(feature_vector_valid)
    if valid_label is not None:
        accuracy=metrics.accuracy_score(valid_label,predictions)
        f1 = metrics.f1_score(valid_label,predictions)
        recall = metrics.recall_score(valid_label,predictions)
        precision=metrics.precision_score(valid_label,predictions)
        auc = metrics.roc_auc_score(valid_label,predictions)
    else:
        accuracy=-1
        f1=-1
        recall=-1
        precision=-1
        auc = -1
    return auc,accuracy,f1,recall,precision,predictions

nb_clf=naive_bayes.MultinomialNB()
logis_clf=linear_model.LogisticRegression()
xgb_clf=xgboost.XGBClassifier()

auc_nb,accuracy_nb ,f1_nb,recall_nb,precision_nb, _ = train_model(nb_clf, xtrain_count.toarray(), train_y, xvalid_count.toarray(),valid_y)
print("NB, Count_vector","auc:",auc_nb,"accuracy: ",accuracy_nb,"f1_score:",f1_nb,"recall_score:",recall_nb,"precision",precision_nb)
auc_lr,accuracy_lr ,f1_lr,recall_lr,precision_lr, _= train_model(logis_clf, xtrain_count.toarray(), train_y, xvalid_count.toarray(),valid_y)
print("LR, Count_vector: ", "auc:",auc_lr,"accuracy: ",accuracy_lr,"f1_score:",f1_lr,"recall_score:",recall_lr,"precision",precision_lr)
auc_xgb,accuracy_xgb ,f1_xgb,recall_xgb,precision_xgb, _= train_model(xgb_clf, xtrain_count.toarray(), train_y, xvalid_count.toarray(),valid_y)
print ("Xgb, Count_vector: ","auc:",auc_xgb,"accuracy: ",accuracy_xgb,"f1_score:",f1_xgb,"recall_score:",recall_xgb,"precision",precision_xgb)


#检验
# _,_ ,_ ,_ ,pred = train_model(logis_clf, xtrain_count.toarray(), train_y, xvalid_count.toarray(),valid_y)
# pred=pd.Series(pred)
# print(pred.value_counts()) #预测没加权重1-1280 0-20检验结果有偏差  加权重1-1104,0-196  实际1-1000 0-300
# result=pd.DataFrame(list(zip(valid_x,valid_y,pred)),columns=['text','true_label','pred_label'],)
# result.to_csv('D:/jupyter_workfile/haizhi/nlp/result/count_lr_pred_valid.csv',encoding="utf_8_sig")  #utf_8_sig是为了解决乱码问题，utf_8也是乱码

#预测
#_ , pred = train_model(logis_clf, xtrain_tfidf.toarray(), train_y, xtest_tfidf.toarray())
#pred=pd.Series(pred)
#print(pred.value_counts())  #1-36928 0-2403检验结果有偏差

#保存文档
#result=pd.DataFrame(list(zip(test_x,pred)),columns=['text','pred_label'],)
#result.to_csv('pred_unlabeled.csv',encoding="utf_8_sig")  #utf_8_sig是为了解决乱码问题，utf_8也是乱码