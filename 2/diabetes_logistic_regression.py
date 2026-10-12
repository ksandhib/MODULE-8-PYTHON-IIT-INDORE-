import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Task 1: Load the dataset using pandas
df = pd.read_csv('diabetes.csv')

# Task 2: Extract data from outcome column as variable Y
Y = df['Outcome']

# Task 3: Extract data from every column except outcome as variable X
X = df.drop('Outcome', axis=1)

# Task 4: Divide the dataset into training and testing in 70% and 30% proportion
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Task 5: Create and train Logistic Regression Model on training set
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Task 6: Make predictions based on the testing set using the trained model
Y_pred = model.predict(X_test)

# Task 7: Check the performance by calculating the confusion matrix and accuracy score of the model
conf_matrix = confusion_matrix(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy Score:", accuracy)
