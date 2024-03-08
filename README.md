# Titanic Survival Prediction

This Python script implements a simple Random Forest Classifier to predict the survival of Titanic passengers. Below are the main functionalities and purposes of the code:

## Functionality and Purpose

### Data Import and Preparation:

- Import data from the CSV file 'titanic.csv,' containing information about Titanic passengers such as passenger class (PClass), age (Age), gender (Gender), and survival status (Survived).
- Calculate the correlation matrix (corr) to understand the correlation between different features.

### Feature Selection:

- Choose three features as input variables (X): passenger class (PClass), age (Age), and gender (Gender).
- Use survival status (Survived) as the target variable (y).

### Data Splitting:

- Split the dataset into a training set (X_train, y_train) and a test set (X_test, y_test).

### Random Forest Model Training:

- Utilize the RandomForestClassifier from the scikit-learn library to build a Random Forest Classifier, with a maximum depth of 3 and a random seed of 0.
- Train the model using the training set (X_train, y_train).

### Model Prediction:

- Use the test set (X_test) to predict outcomes, obtaining predictions (y_pred).

### Performance Evaluation:

- Evaluate the model's performance using a confusion matrix and accuracy score.
- Visualize the performance using a heatmap representation of the confusion matrix.

### Output Predictions:

- Create two new passenger samples, named "Rose 1st 17f" and "Jack 3rd 17m," and predict their outcomes using the trained model.
- Print out the model's accuracy, precision, recall, and confusion matrix metrics.
- Display the predicted outcomes for "Rose" and "Jack."

In summary, this code predicts the survival of Titanic passengers using a Random Forest model, assesses its performance through a confusion matrix and various metrics, and demonstrates the model's predictive capability on new data.

## Usage

**Clone the Repository:**

   ```bash
   git clone https://github.com/xiaosihuangsi/Random-Forest-the-survival-of-passengers-on-the-Titanic.git
   cd titanic-survival-prediction
