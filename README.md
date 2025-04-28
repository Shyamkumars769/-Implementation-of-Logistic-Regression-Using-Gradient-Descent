# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the Logistic Regression Using Gradient Descent

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset and print the values.
3. Define X and Y array and display the value.
4. Find the value for cost and gradient.
5. Plot the decision boundary and predict the Regression value.

## Program:
Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: Shyam Kumar.S

RegisterNumber: 212224040315

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
def gradient_descent(theta, X, y, alpha, num_iterations):
  m = len(y)
  for i in range(num_iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h - y) / m
    theta -= alpha * gradient
  return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5,1, 0)
  return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
```

## Output:
### Dataset
![image](https://github.com/user-attachments/assets/43ef48c5-3399-4c98-8ee6-97b46f21e9ab)

![image](https://github.com/user-attachments/assets/60d9279d-0684-4f3a-9956-aca7d0372025)

![image](https://github.com/user-attachments/assets/82d63c75-64e2-4800-8991-5bdc25daf461)

![image](https://github.com/user-attachments/assets/2c040e70-98c3-4bfd-9f99-36bd27a8b2eb)

### Accuracy and Predicted Values
![image](https://github.com/user-attachments/assets/d1f8fe8b-6ec4-46ff-951f-35583ccc3457)

![image](https://github.com/user-attachments/assets/c6c32c4d-c8fd-4cff-9a37-a396fcb7a71c)

![image](https://github.com/user-attachments/assets/e7f7387d-9e89-454b-9190-17080f2941ee)

![image](https://github.com/user-attachments/assets/43a3bd0b-e99f-4f37-a529-7dee5bbd331c)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

