import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the dataset
df = pd.read_csv('titanic.csv')
corr = df.corr()
X = df.loc[:, ['PClass', 'Age', 'Gender']]
y = df.loc[:, ['Survived']]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Decision Tree Classification model
from sklearn import ensemble
model = ensemble.RandomForestClassifier(max_depth=3, random_state=0)
model.fit(X_train, y_train.values.ravel())

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix and accuracy_score
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
ax = plt.axes()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
ax.set_title("RF")
plt.show()

# Calculate accuracy score
score = accuracy_score(y_test, y_pred)

# Create two example passangers "Rose 1st 17f" and "Jack 3rd 17m"
new_pass = pd.DataFrame([[1, 17, 1], [3,17,0]], columns=['PClass', 'Age', 'Gender'])
y_new = model.predict(new_pass)

print()
print (f'RF acc score: {score:.2f}, precision_score: {tp / (tp + fp):.2f} recall_score: {tp / (tp + fn):.2f} tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}')
print (f'Rose: {y_new[0]}, Jack: {y_new[1]}')





