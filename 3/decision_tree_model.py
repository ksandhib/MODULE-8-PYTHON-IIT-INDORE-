import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Load the dataset using pandas
df = pd.read_csv('diabetes.csv')

# 2. Extract data from outcome column is a variable named Y
Y = df['Outcome']

# 3. Extract data from every column except outcome column in a variable named X
X = df.drop('Outcome', axis=1)

# 4. Divide the dataset into two parts for training and testing in 70% and 30% proportion
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 5. Create and train Decision Tree Model on training set
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

# 6. Make predictions based on the testing set using the trained model
Y_pred = model.predict(X_test)

# 7. Check the performance by calculating the confusion matrix and accuracy score of the model
conf_matrix = confusion_matrix(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy Score:", accuracy)
