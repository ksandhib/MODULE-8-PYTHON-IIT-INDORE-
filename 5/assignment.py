import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Task 1: Load the dataset using pandas
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv('diabetes.csv', header=None, names=column_names)

# Task 2: Extract data from outcome column as Y
Y = data['Outcome']

# Task 3: Extract data from every column except outcome as X
X = data.drop('Outcome', axis=1)

# Task 4: Divide the dataset into training and testing in 70% and 30% proportion
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Task 5: Create and train Naïve Bayes Model on training set
model = GaussianNB()
model.fit(X_train, Y_train)

# Task 6: Make predictions based on the testing set using the trained model
Y_pred = model.predict(X_test)

# Task 7: Check the performance by calculating the confusion matrix and accuracy score of the model
conf_matrix = confusion_matrix(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy Score:")
print(accuracy)
