import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

#DATA MANIPULATION-----------------------------------------------------------------------------------
data=pd.DataFrame(pd.read_csv("A2up.csv"))
data1=pd.DataFrame(pd.read_csv("B2up.csv"))
a=np.zeros((data.shape[0]+data1.shape[0],3))
a[:data.shape[0],:]=data
a[data.shape[0]:,:]=data1
clsa=a[:data.shape[0],:2]
clsb=a[data.shape[0]:,:2]
for i in range(0,a.shape[0]):
   if i<3691:
    a[i,2]=0
   else:
    a[i,2]=1 

#MEAN------------------------------------------
ax=data.mean(axis=0)[0]
ay=data.mean(axis=0)[1]
bx=data1.mean(axis=0)[0]
by=data1.mean(axis=0)[1]
#----------------------------------------------

clus=30
claar=[]
for i in range(0,2*clus):
    if i<clus:
        claar.append(0)
    else:    
        claar.append(1)

Kmean = KMeans(n_clusters=clus)
Kmean.fit(clsa)
U1=Kmean.cluster_centers_
cenar=np.empty((2*clus,3))
for i in range(0,U1.shape[0]):
    cenar[i,:2]=U1[i]

Kmean = KMeans(n_clusters=clus)
Kmean.fit(clsb)
U2=Kmean.cluster_centers_
u2d=pd.DataFrame(U2)
for i in range(0,U2.shape[0]):
    cenar[i+U1.shape[0],:2]=U2[i]

#PLOTING PROTOTYPES------------------------------
cenar[:,2]=claar
color=['black','brown']
poko=['red','blue']
fig = plt.figure(figsize=(28,28)) 
plt.scatter(a[:,0], a[:,1], c=a[:,2], cmap=matplotlib.colors.ListedColormap(poko), marker='o', s=10) 
plt.scatter(cenar[:,0], cenar[:,1], c=cenar[:,2], cmap=matplotlib.colors.ListedColormap(color), marker='s', s=200)
plt.show()
#----------------------------------------------------------

#CHECKING ACCURACY---------------------------------------
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(a[:,:2],a[:,2],test_size=0.4,random_state=4)
from sklearn import metrics

val=26
k_range = range(1,val)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(cenar[:,:2],cenar[:,2])
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
acheck1=max(scores_list)       
#plt.plot(k_range,scores_list)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Testing Accuracy')
#plt.show()
#-----------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(a[:,:2],a[:,2])
p=knn.predict_proba(cenar[:,:2])

#FINDING MIXREGION-------------------------------------
mixcen=[]
for i in range(len(p)):
    if(abs(p[i][0]-p[i][1])<1):
        mixcen.append(i)

mixregion=np.empty((len(mixcen),3))
for i in range(0,len(mixcen)):
    mixregion[i]=cenar[mixcen[i]]
#END--------------------------------------------------------

#PLOTING MIXREGIONS PROTOTYPES--------------------------------
fig = plt.figure(figsize=(28,28)) 
plt.scatter(a[:,0], a[:,1], c=a[:,2], cmap=matplotlib.colors.ListedColormap(poko), marker='o', s=10) 
plt.scatter(mixregion[:,0], mixregion[:,1], c=mixregion[:,2], cmap=matplotlib.colors.ListedColormap(color), marker='s', s=200)
plt.show()
#END------------------------------------

#SHIFTING----------------------------------------------------------
for k in range(0,mixregion.shape[0]):
    if(mixregion[k][2]==0):
        kc=[ax-mixregion[k][0], ay-mixregion[k][1]]
        temp=mixregion[k,:2]
        epsilon=1/100
        while knn.predict_proba(temp.reshape(-1,2))[0][0]<1:
             mixregion[k,:2]=mixregion[k,:2]+(epsilon)* np.asarray(kc)
             epsilon=epsilon+0.02
            
    if(mixregion[k][2]==1):
        kc=[bx-mixregion[k][0], by-mixregion[k][1]]
        epsilon=1/100
        temp=mixregion[k,:2]
        while knn.predict_proba(temp.reshape(-1,2))[0][1]<1 :
            mixregion[k,:2]=mixregion[k,:2]+(epsilon)* np.asarray(kc)
            epsilon=epsilon+0.02        
#END---------------------------------------------------------------------------            

#PLOT OF SHIFT--------------------------------------------------------------------
fig = plt.figure(figsize=(28,28))
plt.scatter(a[:,0], a[:,1], c=a[:,2], cmap=matplotlib.colors.ListedColormap(poko), marker='o', s=10) 
plt.scatter(mixregion[:,0], mixregion[:,1], c=mixregion[:,2], cmap=matplotlib.colors.ListedColormap(color), marker='s', s=200)
plt.show()
#END--------------------------------------------------------------------

#FINAL PLOT ------------------------------------
final=cenar
for i in range(0,len(mixcen)):
    final[mixcen[i]]=mixregion[i]

fig = plt.figure(figsize=(28,28))
plt.scatter(a[:,0], a[:,1], c=a[:,2], cmap=matplotlib.colors.ListedColormap(poko), marker='o', s=10) 
plt.scatter(final[:,0], final[:,1], c=final[:,2], cmap=matplotlib.colors.ListedColormap(color), marker='s', s=150)
plt.scatter(mixregion[:,0], mixregion[:,1], c='green', marker='x', s=800)
plt.show()

#----------------------------------------------------


#FINAL ACCURACY CHECK-------------------------------------------------------------
k_rangei = range(1,val-8)
scoresi = {}
scores_listi = []
from sklearn.utils import shuffle
tesa,tesb=shuffle(a[:,:2],a[:,2])
X_traini,X_testi,y_traini,y_testi = train_test_split(tesa,tesb,test_size=0.5,random_state=4)
for k in k_rangei:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(final[:,:2],final[:,2])
        y_predi=knn.predict(X_testi)
        scoresi[k] = metrics.accuracy_score(y_testi,y_predi)
        scores_listi.append(metrics.accuracy_score(y_testi,y_predi))
acheck2=max(scores_listi)        
plt.plot(k_rangei,scores_listi)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()
#END------------------------------------------------------------------

#ACCURACY CHECK-----------
print(acheck1-acheck2)