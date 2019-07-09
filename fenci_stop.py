import jieba
import re
import os


# train_path="D:/jupyter_workfile/haizhi/nlp/data/train_data.txt"
# valid_path="D:/jupyter_workfile/haizhi/nlp/data/testset.txt"
# test_path="D:/jupyter_workfile/haizhi/nlp/data/unlabeled_data.txt"
stop_path="D:/jupyter_workfile/haizhi/nlp/data/stopwords.txt" #网上载的停用词表+ziji
train_path="D:/jupyter_workfile/haizhi/nlp/data/train_modify.txt"
valid_path="D:/jupyter_workfile/haizhi/nlp/data/testset_modify.txt"
test_path="D:/jupyter_workfile/haizhi/nlp/data/unlabeled_modify.txt"

def train_fenci_stop(data_path,stop_path,po_post,ne_post):
    save=os.getcwd()
    stopwords = [line.strip() for line in open(stop_path, 'r', encoding='utf-8-sig').readlines()]
    pattern1 = re.compile('(.*?)百度快照__label__positive')
    pattern2 = re.compile('(.*?)百度快照__label__negative')
    with open(data_path,"r",encoding='utf8') as f:
        lines=f.readlines()
        for line in lines:
            if 'positive' in line:
                po = re.findall(pattern1,line)
                poline =jieba.cut(po[0],cut_all=False)
                poout1=[]
                for word in poline:
                    if word not in stopwords:
                        if word != '\t':
                           # poout1 += word
                            poout1.append(word)
                poout=' '.join(poout1)
                with open(save+"\\fenci\\"+po_post,'a',encoding='utf8') as pof:
                    pof.write(poout+"\n")
            else:
                ne = re.findall(pattern2,line)
                neline =jieba.cut(ne[0],cut_all=False)
                neout1=[]
                for word in neline:
                    if word not in stopwords:
                        if word != '\t':
                            #neout1 += word  打出来是一个字一个字的，原因不明
                            neout1.append(word)
                neout=' '.join(neout1)
                with open(save+"\\fenci\\"+ne_post,'a',encoding='utf8') as nef:
                    nef.write(neout+"\n")

def test_fenci_stop(data_path,stop_path,post):
    save=os.getcwd()
    stopwords = [line.strip() for line in open(stop_path, 'r', encoding='utf-8-sig').readlines()]
    with open(data_path,"r",encoding='utf8') as unf:
        unlines=unf.readlines()
        for line in unlines:
            un=jieba.cut(line,cut_all=False)
            unout1 = []
            for word in un:
                if word not in stopwords:
                    if word != '\t':
                        unout1.append(word)
            unout = ' '.join(unout1)
            with open(save+"\\fenci\\"+post,'a',encoding='utf8') as unf:
                unf.write(unout)

#train_fenci_stop(train_path,stop_path,po_post="modify_stop_train_po.txt",ne_post="modify_stop_train_ne.txt")
#train_fenci_stop(valid_path,stop_path,po_post="modify_stop_valid_po.txt",ne_post="modify_stop_valid_ne.txt")
#test_fenci_stop(test_path,stop_path,post="modify_stop_unlabel.txt")