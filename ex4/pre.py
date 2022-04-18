import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from skimage import io,transform
import tensorflow as tf
import numpy as np
import glob

flower_dict = {0:'dasiy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

w=100
h=100
c=3
# 全部测试集
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
# 单个文件测试
path1 = "flower_photos_train/daisy/5547758_eea9edfd54_n.jpg"
path2 = "flower_photos_train/dandelion/7355522_b66e5d3078_m.jpg"
path3 = "flower_photos_train/roses/394990940_7af082cf8d_n.jpg"
path4 = "flower_photos_train/sunflowers/6953297_8576bf4ea3.jpg"
path5 = "flower_photos_train/tulips/10791227_7168491604.jpg"
def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

with tf.Session() as sess:

    # data = []
    # data1 = read_one_image(path1)
    # data2 = read_one_image(path2)
    # data3 = read_one_image(path3)
    # data4 = read_one_image(path4)
    # data5 = read_one_image(path5)
    # data.append(data1)
    # data.append(data2)
    # data.append(data3)
    # data.append(data4)
    # data.append(data5)

    test_data,test_label=read_img("flower_photos_test/")

    saver = tf.train.import_meta_graph('model/3thmodel.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:test_data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    pred = sess.run(logits,feed_dict)
    acc_num=0
    for i in range(len(pred)):
        if tf.argmax(pred[i]).eval()==test_label[i]:
            acc_num+=1
    # #打印出预测矩阵
    # print(pred)
    # #打印出预测矩阵每一行最大值的索引
    # print(tf.argmax(pred,1).eval())
    #根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(pred,1).eval()
    for i in range(len(output)):
        print("第",i+1,"朵花预测:"+flower_dict[output[i]],"---- 标签值:",flower_dict[test_label[i]])
    print('总数',len(test_data),"准确数",acc_num,"正确率",acc_num/len(test_data))
