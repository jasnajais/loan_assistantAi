import numpy as np
import pandas as pd
import pickle as pickle
data=pd.read_excel('iris.xls')
data.head(10)
data.info()
data.describe()
x=data[['SL','SW','PL','PW']]
y=data['Classification']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.svm import SVC
sv=SVC()
sv.fit(x_train,y_train)
pickle.dump(sv,open('model.pkl','wb'))