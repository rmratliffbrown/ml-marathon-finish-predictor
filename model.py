import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
marathon = pd.read_csv("marathon.csv")

# Data cleaning
marathon.drop(columns=['Unnamed: 9', 'Unnamed: 10', 'miles4weeks'], inplace=True)
marathon['CrossTraining'] = marathon['CrossTraining'].fillna('None')
marathon['Wall21'] = pd.to_numeric(marathon['Wall21'], errors='coerce')

# Handling missing values
imputer = SimpleImputer(strategy='median')
marathon['Wall21'] = imputer.fit_transform(marathon[['Wall21']])

# Encoding categorical data
categorical_features = ['CrossTraining', 'CATEGORY']
categorical_transformer = OneHotEncoder()

# Apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

X_transformed = preprocessor.fit_transform(marathon.drop('time', axis=1))
y = marathon['time']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Package model 
joblib.dump(model, "regression.pkl")

# Predict on the testing set
#y_pred = model.predict(X_test)

# Performance metrics
#mse = mean_squared_error(y_test, y_pred)
#mae = mean_absolute_error(y_test, y_pred)
#r2_score = model.score(X_test, y_test)

# Output the performance metrics
#print(f'Mean Squared Error: {mse}')
#print(f'Mean Absolute Error: {mae}')
#print(f'R² Score: {r2_score}')


