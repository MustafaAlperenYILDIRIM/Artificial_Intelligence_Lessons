from sklearn.datasets import load_iris
iris=load_iris()

print(iris.feature_names)
print(iris.target_names)
print(iris.target)
print(iris.data)

x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print("Train dataset size",len(x_train))
print("Test dataset size",len(x_test))

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

model.fit(x_train,y_train)
y_prediction=model.predict(x_test)

from sklearn.metrics import confusion_matrix
conf_matrix=confusion_matrix(y_test, y_prediction)
print(conf_matrix)

import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
index = ['setosa','versicolor','virginica'] 
columns = ['setosa','versicolor','virginica'] 
show_matrix = pd.DataFrame(conf_matrix,columns,index) 
plt.figure(figsize=(10,6)) 
sns.heatmap(show_matrix,annot=True) 