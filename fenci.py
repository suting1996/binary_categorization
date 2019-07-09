import jieba
import re

train_path="D:/jupyter_workfile/haizhi/nlp/data/train_data.txt"
valid_path="D:/jupyter_workfile/haizhi/nlp/data/testset.txt"
pred_path="D:/jupyter_workfile/haizhi/nlp/data/unlabeled_data.txt"
pattern1 = re.compile('(.*?)百度快照__label__positive')
pattern2 = re.compile('(.*?)百度快照__label__negative')
rex=re.compile('[，,。.？?/!%*:;‘•’·`【】《》（）()<>、""“”…—__+-×◎]')##去符号
with open(train_path,"r",encoding='utf8') as f:
    lines=f.readlines()
    for line in lines:
        if 'positive' in line:
            po = re.findall(pattern1,line)
            poline =jieba.cut(po[0],cut_all=False)
            poout=' '.join(poline).replace('-','')
            poout=re.sub(rex,' ',poout)
            with open("D:/jupyter_workfile/haizhi/nlp/data/po_train.txt",'a',encoding='utf8') as po:
                po.write(poout+"\n")
        else:
            ne = re.findall(pattern2,line)
            neline =jieba.cut(ne[0],cut_all=False)
            neout=' '.join(neline).replace('-','')
            neout=re.sub(rex,' ',neout)
            with open("D:/jupyter_workfile/haizhi/nlp/data/ne_train.txt",'a',encoding='utf8') as ne:
                ne.write(neout+"\n")

with open(valid_path,"r",encoding='utf8') as vf:
    vlines=vf.readlines()
    for line in vlines:
        if 'positive' in line:
            vpo = re.findall(pattern1,line)
            vpoline =jieba.cut(vpo[0],cut_all=False)
            vpoout=' '.join(vpoline).replace('-','')
            vpoout=re.sub(rex,' ',vpoout)
            with open("D:/jupyter_workfile/haizhi/nlp/data/po_valid.txt",'a',encoding='utf8') as vpo:
                vpo.write(vpoout+"\n")
        else:
            vne = re.findall(pattern2,line)
            vneline =jieba.cut(vne[0],cut_all=False)
            vneout=' '.join(vneline).replace('-','')
            vneout=re.sub(rex,' ',vneout)
            with open("D:/jupyter_workfile/haizhi/nlp/data/ne_valid.txt",'a',encoding='utf8') as vne:
                vne.write(vneout+"\n")

with open(pred_path,"r",encoding='utf8') as unf:
    unlines=unf.readlines()
    for line in unlines:
        un=jieba.cut(line,cut_all=False)
        un=' '.join(un).replace('-','')
        un=re.sub(rex,' ',un)
        with open("D:/jupyter_workfile/haizhi/nlp/data/unlabeled_out.txt",'a',encoding='utf8') as unf:
            unf.write(un+"\n")