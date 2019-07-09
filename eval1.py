# coding:utf-8

import tensorflow as tf
import numpy as np
import os
import pandas as pd
import data_help
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn import metrics

# ============
# 无分词命令行执行：python eval.py --eval_train --checkpoint_dir="./runs/1547716858/checkpoints/"
#   分词命令行执行：python eval.py --eval_train --checkpoint_dir="./runs/1547734140/checkpoints/"
#修正数据集执行：python eval.py --eval_train --checkpoint_dir="./runs/1547794934/checkpoints/"
# # # ============

# 参数：我们这里使用命令行传入参数的方式执行该脚本

# tf.flags.DEFINE_string("positive_file", "D:/jupyter_workfile/haizhi/po_valid.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_file", "D:/jupyter_workfile/haizhi/ne_valid.txt", "Data source for the negative data.")
# tf.flags.DEFINE_string("positive_file", "D:/jupyter_workfile/haizhi/nlp/fenci/stop_valid_po.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_file", "D:/jupyter_workfile/haizhi/nlp/fenci/stop_valid_ne.txt", "Data source for the negative data.")
tf.flags.DEFINE_string("positive_file", "D:/jupyter_workfile/haizhi/nlp/fenci/modify_stop_valid_po.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_file", "D:/jupyter_workfile/haizhi/nlp/fenci/modify_stop_valid_ne.txt", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# 填写训练获得模型的存储位置
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1555659313/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train",True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement",False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_help.load_data_and_labels(FLAGS.positive_file, FLAGS.negative_file)
    y_test = np.argmin(y_test, axis=1)
else:
    predout_path = "D:/jupyter_workfile/haizhi/nlp/fenci/stop_unlabel.txt"
    x1 = list(open(predout_path, "r",encoding='utf8').readlines())
    x_raw = [s.strip() for s in x1]
    y_test = []
# Map data into vocabulary
print(y_test)
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print("checkpoint_file========", checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_help.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # 存储模型预测结果
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
        print(all_predictions)
# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    accuracy=correct_predictions / float(len(y_test))
    f1 = metrics.f1_score(y_test, all_predictions)
    recall = metrics.recall_score(y_test,all_predictions)
    precision = metrics.precision_score(y_test, all_predictions)
    auc =  metrics.roc_auc_score(y_test, all_predictions)
    print("Total number of test examples: {}".format(len(y_test)))
    #print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
    print("TextCNN, embedding ", "auc:",auc,"accuracy: ", accuracy, "f1_score:", f1, "recall_score:", recall,"precision", precision)


# 将二分类的0，1标签转化为中文标签
y = []
for i in all_predictions:
     if i == 1.0:
         y.append("positive")
     else:
        y.append("negative")
 # 把预测的结果保存到本地
predictions_human_readable = np.column_stack((y,y_test, np.array(x_raw)))
#out_path = "D:/jupyter_workfile/haizhi/nlp/result/pred_textCnn_valid.csv"
out_path = "D:/jupyter_workfile/haizhi/nlp/result/modify_pred_stop_textCnn_valid1.csv"
print("Saving evaluation to {0}".format(out_path))
pd.DataFrame(predictions_human_readable,columns=['pred_label','true_label','text']).to_csv(out_path,encoding="utf_8_sig")
