#!/usr/bin/env python
# coding: utf-8

# In[13]:


# import libraries
import numpy as np
import mahotas
import cv2
import os
import h5py
from skimage.feature import hog
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# In[2]:


def processing_color_histogram(image):
    hist  = cv2.calcHist([image], [0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def processing_hog(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(image)

def processing_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray).mean(axis=0)


# In[20]:


img_size = tuple((256,256))
hog_size=tuple((100,100))
test_size = 0.10 # train_test_split size

def process_image(image):
    resized_image = cv2.resize(image, img_size)
    resized_hog= cv2.resize(image, hog_size)
    blur_img = cv2.GaussianBlur(resized_image, (5, 5), 0)
    color_historgram_vector  = processing_color_histogram(blur_img)
    haralick_vector = processing_haralick(blur_img)
    hstack = np.hstack([color_historgram_vector, haralick_vector])
    hog_vector = processing_hog(resized_hog)
    return hstack,hog_vector

def normalize_features(features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(features)


# In[4]:


import os
path = 'train'
label=[]
global_col=[]
global_hog=[]
for folder in os.listdir(path):
    if folder!=".DS_Store":
        imgpath=os.path.join(path,folder)
        for filename in os.listdir(imgpath):
            if filename !=".DS_Store":
                img=cv2.imread(os.path.join(imgpath, filename))
                c=[]
                l=[]
                (c,l)=process_image(img)
                global_col.append(c)
                global_hog.append(l)
                label.append(folder)


# In[15]:


from sklearn.preprocessing import LabelEncoder
targetNames=np.unique(label)
le=LabelEncoder()
target=le.fit_transform(label)


# In[6]:


h5f_data=h5py.File("col.h5","w")
h5f_data.create_dataset("dataset_1",data=np.array(global_col))

h5f_data.close()

h5f_data=h5py.File("hog.h5","w")
h5f_data.create_dataset("dataset_1",data=np.array(global_hog))

h5f_data.close()


# In[14]:


h5f_data=h5py.File("col.h5","r")
global_cold=h5f_data["dataset_1"]
global_col=np.array(global_cold)
h5f_data.close()

h5f_data=h5py.File("hog.h5","r")
global_hogd=h5f_data["dataset_1"]
global_hog=np.array(global_hogd)
h5f_data.close()

h5f_data=h5py.File("label.h5","r")
global_l=h5f_data["dataset_1"]
label=np.array(global_l)
h5f_data.close()


# In[16]:


index=np.arange(0,len(target))
(train_index, test_index, train_labels, test_labels) = train_test_split(index,target, test_size=0.15)


# In[17]:


train_hog=[]
train_col=[]
test_hog=[]
test_col=[]
for i in train_index:
    train_hog.append(global_hog[i])
    train_col.append(global_col[i])
for i in test_index:
    test_hog.append(global_hog[i])
    test_col.append(global_col[i])


# In[ ]:


nsvdknnf1=[]
nsvdrff1=[]
nsvdlf1=[]
nsvdknna=[]
nsvdrfa=[]
nsvdla=[]
for nsvd in (30,50,80,100,200,300,400,500):
    svd = TruncatedSVD(n_components=nsvd)
    svd.fit(train_hog)
    train_hogs_svd = svd.transform(train_hog)
    test_hogs_svd = svd.transform(test_hog)
    global_train = np.hstack([np.array(train_col), np.array(train_hogs_svd)])
    global_test = np.hstack([np.array(test_col), np.array(test_hogs_svd)])
    normalized_train_features = normalize_features(global_train)
    normalized_test_features = normalize_features(global_test)
    for nknn in (5,8,10,15,20,25,30,50):
        clf  = KNeighborsClassifier(n_neighbors=nknn)
        clf.fit(normalized_train_features, train_labels)
        prediction = clf.predict(normalized_test_features)
        print(nsvd,nknn)
        w=f1_score(test_labels, prediction, average='weighted')
        a=accuracy_score(test_labels, prediction)
        nsvdknnf1.append([nsvd,nknn,w])
        nsvdknna.append([nsvd,nknn,a])
        print("weighted f1 score %f " % w)
        print("accuracy %f " % a)
    for nrf in (80,100,200,300):
        print("rf ",nsvd,nrf)
        clf  = RandomForestClassifier(n_estimators=nrf)
        clf.fit(normalized_train_features, train_labels)
        prediction = clf.predict(normalized_test_features)
        w=f1_score(test_labels, prediction, average='weighted')
        a=accuracy_score(test_labels, prediction)
        nsvdrff1.append([nsvd,nrf,w])
        nsvdrfa.append([nsvd,nrf,a])
        print("weighted f1 score %f " % w)
        print("accuracy %f " % a)
    clf = LogisticRegression(random_state=7, solver='lbfgs',multi_class='multinomial').fit(normalized_train_features, train_labels)
    prediction = clf.predict(normalized_test_features)
    print("lr")
    w= f1_score(test_labels, prediction, average='weighted')
    a=accuracy_score(test_labels, prediction)
    nsvdlf1.append([nsvd,w])
    nsvdla.append([nsvd,a])
    print("weighted f1 score %f " %w)
    print("accuracy %f " % a)


# In[24]:


nsvdknnf1=[]
nsvdrff1=[]
nsvdlf1=[]
nsvdknna=[]
nsvdrfa=[]
nsvdla=[]
nsvd=80
svd = TruncatedSVD(n_components=nsvd)
svd.fit(train_hog)
train_hogs_svd = svd.transform(train_hog)
test_hogs_svd = svd.transform(test_hog)
global_train = np.hstack([np.array(train_col), np.array(train_hogs_svd)])
global_test = np.hstack([np.array(test_col), np.array(test_hogs_svd)])
normalized_train_features = normalize_features(global_train)
normalized_test_features = normalize_features(global_test)
clf = LogisticRegression(random_state=7, solver='lbfgs',multi_class='multinomial').fit(normalized_train_features, train_labels)
prediction = clf.predict(normalized_test_features)
print("lr")
w= f1_score(test_labels, prediction, average='weighted')
a=accuracy_score(test_labels, prediction)
nsvdlf1.append([nsvd,w])
nsvdla.append([nsvd,a])
print("weighted f1 score %f " %w)
print("accuracy %f " % a)


# In[11]:


nsvdknnf1=[]
nsvdrff1=[]
nsvdlf1=[]
nsvdknna=[]
nsvdrfa=[]
nsvdla=[]

    
global_train = np.hstack([np.array(train_col), np.array(train_hog)])
global_test = np.hstack([np.array(test_col), np.array(test_hog)])
normalized_train_features = normalize_features(global_train)
normalized_test_features = normalize_features(global_test)

for nrf in (100,200,300,400):
    print("rf ",nrf)
    clf  = RandomForestClassifier(n_estimators=nrf)
    clf.fit(normalized_train_features, train_labels)
    prediction = clf.predict(normalized_test_features)
    w=f1_score(test_labels, prediction, average='weighted')
    a=accuracy_score(test_labels, prediction)
    nsvdrff1.append([nrf,w])
    nsvdrfa.append([nrf,a])
    print("weighted f1 score %f " % w)
    print("accuracy %f " % a)

clf = LogisticRegression(random_state=7, solver='lbfgs',multi_class='multinomial').fit(normalized_train_features, train_labels)
prediction = clf.predict(normalized_test_features)
print("lr")
w= f1_score(test_labels, prediction, average='weighted')
a=accuracy_score(test_labels, prediction)
nsvdlf1.append(w)
nsvdla.append(a)
print("weighted f1 score %f " %w)
print("accuracy %f " % a)   
    
for nknn in (5,10,15,20):
    clf  = KNeighborsClassifier(n_neighbors=nknn)
    clf.fit(normalized_train_features, train_labels)
    prediction = clf.predict(normalized_test_features)
    print(nknn)
    w=f1_score(test_labels, prediction, average='weighted')
    a=accuracy_score(test_labels, prediction)
    nsvdknnf1.append([nknn,w])
    nsvdknna.append([nknn,a])
    print("weighted f1 score %f " % w)
    print("accuracy %f " % a)


# In[ ]:





# In[15]:


for npca in (30,50,80,100,200,300,400,500):
    pca = PCA(n_components=npca)
    pca.fit(train_hog)
    train_hogs_pca = pca.transform(train_hog)
    test_hogs_pca = pca.transform(test_hog)
    global_train = np.hstack([np.array(train_col), np.array(train_hogs_pca)])
    global_test = np.hstack([np.array(test_col), np.array(test_hogs_pca)])
    normalized_train_features = normalize_features(global_train)
    normalized_test_features = normalize_features(global_test)
    
    for nrf in (80,100,200,300,400):
        print("rf  ",npca,nrf)
        clf  = RandomForestClassifier(n_estimators=nrf, random_state=4)
        clf.fit(normalized_train_features, train_labels)
        prediction = clf.predict(normalized_test_features)
        print("weighted f1 score %f " % f1_score(test_labels, prediction, average='weighted'))
        print("macro f1 score %f " % f1_score(test_labels, prediction, average='macro'))
        print("micro f1 score %f " % f1_score(test_labels, prediction, average='micro'))
        print("accuracy %f " % accuracy_score(test_labels, prediction))
    for nknn in (5,10,15,20):
        clf  = KNeighborsClassifier(n_neighbors=nknn)
        clf.fit(normalized_train_features, train_labels)
        prediction = clf.predict(normalized_test_features)
        print("KNN ",npca,nknn)
        print("weighted f1 score %f " % f1_score(test_labels, prediction, average='weighted'))
        print("macro f1 score %f " % f1_score(test_labels, prediction, average='macro'))
        print("micro f1 score %f " % f1_score(test_labels, prediction, average='micro'))
        print("accuracy %f " % accuracy_score(test_labels, prediction))
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(normalized_train_features, train_labels)
    prediction = clf.predict(normalized_test_features)
    print("lr")
    print("weighted f1 score %f " % f1_score(test_labels, prediction, average='weighted'))
    print("macro f1 score %f " % f1_score(test_labels, prediction, average='macro'))
    print("micro f1 score %f " % f1_score(test_labels, prediction, average='micro'))
    print("accuracy %f " % accuracy_score(test_labels, prediction))


# In[ ]:


for npca in (50,80,100,200,300,400,500):
    pca = PCA(n_components=npca)
    pca.fit(train_hog)
    train_hogs_pca = pca.transform(train_hog)
    test_hogs_pca = pca.transform(test_hog)
    global_train = np.hstack([np.array(train_col), np.array(train_hogs_pca)])
    global_test = np.hstack([np.array(test_col), np.array(test_hogs_pca)])
    normalized_train_features = normalize_features(global_train)
    normalized_test_features = normalize_features(global_test)
    
    for nrf in (80,100,200,300,400):
        print("rf  ",npca,nrf)
        clf  = RandomForestClassifier(n_estimators=nrf, random_state=4)
        clf.fit(normalized_train_features, train_labels)
        prediction = clf.predict(normalized_test_features)
        print("weighted f1 score %f " % f1_score(test_labels, prediction, average='weighted'))
        print("macro f1 score %f " % f1_score(test_labels, prediction, average='macro'))
        print("micro f1 score %f " % f1_score(test_labels, prediction, average='micro'))
        print("accuracy %f " % accuracy_score(test_labels, prediction))
    


# In[78]:


import matplotlib.pyplot as plt

data=[[0.05379,0.05452,0.0502],[0.06325,0.0580,0.05347],[0.06089,0.05867,0.05347],[0.0616,0.0591,0.0533],[0.0857,0.0847,0.0764],[0.103,0.1029,0.0954],[0.112,0.1129,0.1059],[0.06,0.0744,0.07627]]
fig, ax = plt.subplots()
fig.set_size_inches(18, 5, forward=True)
plt.title('KNN vs RF vs LR')
plt.ylabel('Accuracy')
ax.boxplot(data)
ax.set_xticklabels(["k=5","k=10","k=15","k=20","n=100","n=200","n=300","LR"])

plt.show()


# In[92]:


import matplotlib.pyplot as plt

data=[[0.05379,0.05452,0.0502],[0.06325,0.0580,0.05347],[0.06089,0.05867,0.05347],[0.0616,0.0591,0.0533],[0.049,0.055,0.055],[0.047,0.052,0.049],[0.042,0.051,0.045],[0.041,0.047,0.0449]]
fig, ax = plt.subplots()
fig.set_size_inches(9, 5, forward=True)
plt.title('KNN')
plt.ylabel('Accuracy')
ax.boxplot(data)
ax.set_xticklabels(["k=5","k=10","k=15","k=20","k=2","k=4","k=6","k=8"])
plt.ylim(0, 0.12)
plt.show()


# In[99]:


data=[[0.0857,0.0847,0.0764],[0.103,0.1029,0.0954],[0.112,0.1129,0.1059],[0.082,0.088,0.0742],[0.092,0.099,0.0899],[0.099,0.097,0.0967]]
fig, ax = plt.subplots()
fig.set_size_inches(6, 5, forward=True)
plt.title('RF')
#plt.ylabel('Accuracy')
ax.boxplot(data)
ax.set_xticklabels(["n=100","n=200","n=300","n=100","n=200","n=300"])
plt.ylim(0, 0.12)
plt.yticks([])
plt.show()


# In[100]:


data=[[0.06,0.0744,0.07627],[0.0723,0.0654,0.067]]
fig, ax = plt.subplots()
fig.set_size_inches(6, 5, forward=True)
plt.title('LR')
#plt.ylabel('Accuracy')
ax.boxplot(data)
ax.set_xticklabels(["LR","LR"])
plt.ylim(0, 0.12)
plt.yticks([])
plt.show()


# In[ ]:




