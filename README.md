# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Adhithya.S
RegisterNumber:  212222240003
*/
import pandas as pd
data=pd.read_csv('/Placement_Data(1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)#Accuracy Score = (TP+TN)/(TP+FN+TN+FP)
#accuracy_score(y_true,y_pred,normalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion #11+24=35 -correct predictions,5+3=8 incorrect predictions

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:
### Placement data
![Screenshot 2023-10-10 091338](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/ecb46412-3cf6-4343-8fb9-37a7e2d8f06c)

### Salary data
![Screenshot 2023-10-10 091351](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/5cb34c78-d5d2-4de3-94fa-e7d824727acf)

### Checking the null() function
![Screenshot 2023-10-10 091418](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/5d23df91-f30e-45f6-b9db-170691228d30)


### Data Duplicate
![Screenshot 2023-10-10 091431](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/d66d6e1a-92e5-4302-a808-5495d954edaa)


### Print data
![Screenshot 2023-10-10 091451](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/27e4ec3b-f5c7-4737-b009-22ac0b4719ec)


### Data-Status
![Screenshot 2023-10-10 091502](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/ca34485b-3d94-4c60-a3fd-0b56c823a418)

### y_prediction array
![Screenshot 2023-10-10 091519](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/165bfb98-44db-4c5d-b96b-53016bebab45)

### Accuracy value
![Screenshot 2023-10-10 091529](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/3296dc9d-161f-4635-80aa-7f6f83b1197d)

### Confusion array
![Screenshot 2023-10-10 091538](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/b52356e8-1b01-49ec-86b6-622c11acaa58)

### Classification report
![Screenshot 2023-10-10 091548](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/1350db60-1950-4fee-a70b-f2eca75a6edb)

### Prediction of LR
![Screenshot 2023-10-10 091607](https://github.com/s-adhithya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497423/bc327dbf-a964-4560-8463-bbb51f04bfb1)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
