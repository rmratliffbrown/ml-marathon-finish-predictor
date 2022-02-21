import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

marathon = pd.read_csv("marathon.csv")

X = marathon[["miles4weeks", "speed4week"]]
y = marathon["time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109)

reg.fit(X_train, y_train)

import joblib
joblib.dump(reg, "reg.pkl")