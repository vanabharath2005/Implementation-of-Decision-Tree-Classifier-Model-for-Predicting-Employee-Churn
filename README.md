
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
 
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: vana bharath.D
RegisterNumber:212223040231 
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![image](https://github.com/user-attachments/assets/748a201c-3f13-4357-b2ed-9b6dea765899)
![image](https://github.com/user-attachments/assets/389167e1-24d1-4dea-be90-fc8adafb902e)
![image](https://github.com/user-attachments/assets/559e1089-bc55-43e8-86e7-820e87e5f002)
![image](https://github.com/user-attachments/assets/79678a6d-7525-4932-b412-74518439fa03)
![image](https://github.com/user-attachments/assets/6c6b3807-a43a-41d6-a08e-7c9954d3b05b)
![image](https://github.com/user-attachments/assets/2d8bb0fc-53b0-4210-88ed-21d3cabe6747)
![image](https://github.com/user-attachments/assets/174b4377-a0ab-4c19-b384-74effd6fa505)
![image](https://github.com/user-attachments/assets/0f864488-54ee-47e8-8db0-7621b2bf1f2a)















## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
