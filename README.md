# Customer-Segmentation-for-MALL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("Mall_Customers.csv")

df.head()

df.info()

df.shape

df.describe()

col=df.select_dtypes("int64").columns

for i in df[col]:
    print(sns.distplot(df[i]))
    plt.show()

def box_plot(df,col):
    df.boxplot(col)
    plt.show()

for i in df[col]:
    print(i)
    box_plot(df,i)
    plt.show()

q1=df["Annual Income (k$)"].quantile(0.25)
q3=df["Annual Income (k$)"].quantile(0.75)
iqr=q3-q1

dt=df[(df["Annual Income (k$)"]<q1-1.5*iqr)|(df["Annual Income (k$)"]>q3+1.5*iqr)]

dt

df.drop([198,199],axis=0,inplace=True)

sns.boxplot(df["Annual Income (k$)"])

from scipy.stats import skew

sns.countplot(df["Gender"])

age_18_25=df.Age[(df.Age>=18)&(df.Age<=25)]
age_26_35=df.Age[(df.Age>=26)&(df.Age<=35)]
age_36_45=df.Age[(df.Age>=36)&(df.Age<=45)]
age_46_55=df.Age[(df.Age>=46)&(df.Age<=55)]
age_55above=df.Age[df.Age>=56]

agex=["18-25","26-35","36-45","46-55","55+"]
agey=[len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_55above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=agex,y=agey,palette="mako")
plt.title("number of customers and Ages")
plt.xlabel("age")
plt.ylabel("number of customers")
plt.show()

sns.relplot(x="Annual Income (k$)",y="Spending Score (1-100)",data=df)

ss_1_20=df.Age[(df["Spending Score (1-100)"]>=1)&(df["Spending Score (1-100)"]<=20)]
ss_21_40=df.Age[(df["Spending Score (1-100)"]>=21)&(df["Spending Score (1-100)"]<=40)]
ss_41_60=df.Age[(df["Spending Score (1-100)"]>=41)&(df["Spending Score (1-100)"])<=60]
ss_61_80=df.Age[(df["Spending Score (1-100)"]>=61)&(df["Spending Score (1-100)"])<=80]
ss_81_100=df.Age[(df["Spending Score (1-100)"]>=81)&(df["Spending Score (1-100)"])<=100]

ssx=["1-10","21-40","41-60","61-80","81-100"]
ssy=[len(ss_1_20.values),len(ss_21_40.values),len(ss_41_60.values),len(ss_61_80.values),len(ss_81_100.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=ssx,y=ssy,palette="rocket")
plt.title("Spenfing Score")
plt.xlabel("Score")
plt.ylabel("number of customers having the score")
plt.show()

x1=df.loc[:,["Annual Income (k$)","Spending Score (1-100)"]].values

from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(x1)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color="red",marker="*")
plt.show()

kmeans= KMeans(n_clusters= 5,init="k-means++", random_state=0)
y_kmeans=kmeans.fit_predict(x1)


y_kmeans

df["ykmeans"]=y_kmeans

plt.scatter(x1[:,0],x1[:,1],c=y_kmeans,s=100,cmap="rainbow")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='black',label="Centroids")

plt.xlabel('annual income')
plt.ylabel("spending score")
plt.title('clusters of customers')
plt.legend()
plt.show()

df

x=df.iloc[:,3:5]

y=df["ykmeans"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1,random_state=3)

def train_model(model):
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtrain)
    
    train_score=model.score(xtrain,ytrain)
    test_score=model.score(xtest,ytest)
    
    print("train_score:",train_score,"\n","test_score:",test_score)
    return model
   

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.metrics import classification_report

gnb=train_model(GaussianNB())

from sklearn.tree import DecisionTreeClassifier

dt=train_model(DecisionTreeClassifier())

x

