#!/usr/bin/env python3
# -*- coding: utf-8 
"""
Created on Sun May 13 22:26:59 2018

@author: zhanglei
"""
from sklearn import datasets,decomposition,manifold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import cv2
import os
import numpy as np




flag=False   #flag==Ture:pca  flag==false:lle
    



#训练集组成： AT&T  LFW
#测试集组成： LFW

#提取AT&T数据  并做预处理（resize bgr2gray）返回数据集和标签
def prepare_at_data():
    at_images_dir='/home/zhanglei/face_classification/ORL/'
    names=os.listdir(at_images_dir)
    at_labels=[int((x.split('_')[0]).split('s')[1]) for x in names]
    names=[at_images_dir+x for x in names]
    at_X=np.zeros((1,37*50))
    for i in range(len(names)):  
        image=cv2.imread(names[i])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        new_imge=cv2.resize(image,(37,50))
        temp_data=np.reshape(new_imge,(1,37*50))
        at_X=np.concatenate((at_X,temp_data),axis=0) 
           
    return at_X[1:,:],at_labels
    

at_X,at_labels=prepare_at_data() 


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)#抽取LFW数据


n_samples, h, w = lfw_people.images.shape


X = lfw_people.data
n_features = X.shape[1]


y = lfw_people.target
y=[t+41 for t in y]        #LFW label
target_names = lfw_people.target_names  
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


#划出一部分LFW数据集加入训练集 另外一部分作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



#将AT&T 数据集和LFW数据集的一部分组合起来 构成训练集  得到相应标签
X_train=np.concatenate((X_train,at_X),axis=0)
final_label=[]
for i in range(len(y_train)):
    final_label.append(y_train[i])
for i in range(len(at_labels)):
    final_label.append(at_labels[i])
y_train=final_label




n_components = 150                  #降维后的维度数目
k=100



kf = KFold(n_splits=5, shuffle=True)
cv_accuracy=[]
gama=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]    

for i in range(len(gama)):
    t=1
    para=gama[i]
    temp_accuracy=0
    for train, cv in kf.split(X_train, y_train):
        print('gama:',para,'kf:',t) 
        X_tmp_train=np.array(X_train)[train]
        y_tmp_train=np.array(y_train)[train]
        X_cv=np.array(X_train)[cv]
        y_cv=np.array(y_train)[cv]       #在训练集中进一步划分出验证集
        if flag==True:
            pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_tmp_train)
            X_train_reduced = pca.transform(X_tmp_train)
            X_cv_reduced = pca.transform(X_cv)
        if flag==False:
            lle=manifold.LocallyLinearEmbedding(n_components=n_components,n_neighbors=k)
            X_train_reduced=lle.fit_transform(X_tmp_train)
            X_cv_reduced = lle.transform(X_cv)
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',gamma=para), param_grid)
        clf = clf.fit(X_train_reduced, y_tmp_train)
        y_cv_pred = clf.predict(X_cv_reduced)
        temp_accuracy=temp_accuracy+( sum(y_cv_pred==y_cv)/len(y_cv) )
        t=t+1
    cv_accuracy.append(temp_accuracy/5)  #每个参数在验证集上的准确度
        


plt.figure(1)
plt.plot(gama, cv_accuracy)
plt.xlabel("Gamma")
plt.ylabel("Precision")
plt.title('The accuracy of cv with Different values of Gamma')
plt.legend()
plt.show()





#test
#最佳gama为 0.001
if flag==True:
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    X_train_reduced = pca.transform(X_train)
    X_test_reduced = pca.transform(X_test)
if flag==False:
    lle=manifold.LocallyLinearEmbedding(n_components=n_components,n_neighbors=k)
    X_train_reduced=lle.fit_transform(X_train)
    X_test_reduced = lle.transform(X_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',gamma=0.001), param_grid)
clf = clf.fit(X_train_reduced, y_train)
y_pred = clf.predict(X_test_reduced)
test_accuracy=sum(y_pred==y_test)/len(y_test) 




