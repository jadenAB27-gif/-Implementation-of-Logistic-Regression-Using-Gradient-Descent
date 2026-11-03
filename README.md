# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries and load the Placement_Data.csv dataset.
2. Select the academic performance columns as input features and convert the placement status into numerical values.
3. Split the dataset into training and testing sets, and normalize the input features.
4. Initialize the weights and bias, apply Gradient Descent to train the Logistic Regression model, and predict the placement status.

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Jaden Samuel Abraham
RegisterNumber: 212225040138

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv("Placement_Data.csv")
X=data[['ssc_p','hsc_p','degree_p','etest_p','mba_p']].values
y=data['status'].map({'Placed':1,'Not Placed':0}).values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
weights=np.zeros(X_train.shape[1])
bias=0
learning_rate=0.1
epochs=10000
for i in range(epochs):
    z=np.dot(X_train,weights)+bias
    z=np.clip(z,-500,500)
    predictions=1/(1+np.exp(-z))
    dw=np.dot(X_train.T,(predictions-y_train))/len(y_train)
    db=np.mean(predictions-y_train)
    weights-=learning_rate*dw
    bias-=learning_rate*db
z=np.dot(X_test,weights)+bias
predictions=1/(1+np.exp(-z))
y_pred=(predictions>=0.5).astype(int)
print("Actual Placement Status:",y_test)
print("Predicted Placement Status:",y_pred)
print("Accuracy:",accuracy_score(y_test,y_pred)*100,"%")
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
student=np.array([[75,80,70,85,78]])
student=scaler.transform(student)
z=np.dot(student,weights)+bias
probability=1/(1+np.exp(-z))
print("Placement Probability:",probability[0]*100,"%")
if probability[0]>=0.5:
    print("Student is Placed")
else:
    print("Student is Not Placed") 

```

## Output:

<img width="1301" height="572" alt="image" src="https://github.com/user-attachments/assets/19cccc3d-cc70-4172-aa55-1539f1a1a058" />
<img width="1531" height="701" alt="image" src="https://github.com/user-attachments/assets/146a83c0-b363-48b6-bb04-d9ec95b7bf2f" />
<img width="991" height="215" alt="image" src="https://github.com/user-attachments/assets/d85038dd-43a1-48b7-874c-571e1af9285f" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

