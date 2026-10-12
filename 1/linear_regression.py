import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Load the dataset using pandas
data = pd.read_csv('data.csv')

# 2. Extract data from years experience column is a variable named X
X = data[['YearsExperience']]

# 3. Extract data from the salary column is a variable named Y
Y = data['Salary']

# 4. Divide the dataset into two parts for training and testing in 66% and 33% proportion
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# 5. Create and train Linear Regression Model on training set
model = LinearRegression()
model.fit(X_train, Y_train)

# 6. Make predictions based on the testing set using the trained model
Y_pred = model.predict(X_test)

# 7. Check the performance by calculating the r2 score of the model
r2 = r2_score(Y_test, Y_pred)
print(f"R² Score: {r2}")
