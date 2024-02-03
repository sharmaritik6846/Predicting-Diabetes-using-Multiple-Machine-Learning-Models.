# Predicting-Diabetes-using-Multiple-Machine-Learning-Models.

From our conversation, we've discussed the following key points:

1. **K-Nearest Neighbors (KNN) Model**: We've discussed how to train a K-Nearest Neighbors (KNN) model for predicting diabetes based on various health parameters. The model was trained and its performance was evaluated using accuracy score, confusion matrix, and classification report.

2. **Model Prediction**: We've created a function `predict_diabetes` that takes a trained model and uses it to predict whether a patient has diabetes based on their health parameters. The function prompts the user to enter their health parameters, preprocesses the input data, and uses the model to make a prediction.

3. **Model Saving and Loading**: We've discussed how to save a trained model to a file and then load it back into memory using the `pickle` module in Python. This allows us to reuse the model without having to retrain it every time.

4. **Model Performance**: We've discussed potential reasons why the model might not be making accurate predictions, such as issues with the model's performance, the quality of the data, the importance of the features, and the limitations of the KNN algorithm.

5. **Improving Model Performance**: We've discussed several strategies for improving the model's performance, including checking the data, checking the model, tuning the model, and handling class imbalance.

Remember, machine learning is an iterative process. It often takes several rounds of refining the data and the model to get accurate predictions. 😊

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

data1 = pd.read_csv("/content/diabetes.csv")
data1.head()

data1.tail()

data1.shape

data1.describe()

data1.info()

data1.isna()

data1.isna().sum()

Description=data1.describe().T
Description

features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

for i in range(0, len(features), 2):
    fig = plt.figure(figsize=(15,5)) 
    plt.subplot(1,2,1)
    sns.boxplot(x=features[i], data=data1)
    
if i+1 < len(features):
        plt.subplot(1,2,2)
        sns.boxplot(x=features[i+1], data=data1)
    
plt.show()

    
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(data1['Age'], bins = 20, ax=ax[0,0]) 
sns.distplot(data1['Pregnancies'], bins = 20, ax=ax[0,1]) 
sns.distplot(data1['Glucose'], bins = 20, ax=ax[1,0]) 
sns.distplot(data1['BloodPressure'], bins = 20, ax=ax[1,1]) 
sns.distplot(data1['SkinThickness'], bins = 20, ax=ax[2,0])
sns.distplot(data1['Insulin'], bins = 20, ax=ax[2,1])
sns.distplot(data1['DiabetesPedigreeFunction'], bins = 20, ax=ax[3,0]) 
sns.distplot(data1['BMI'], bins = 20, ax=ax[3,1])
plt.tight_layout()
plt.show()

sns.pairplot(data = data1, hue = 'Outcome')
plt.show()

data1.hist(bins=50, figsize=(20,15))
plt.show()

fig, ax = plt.subplots(figsize = (12,10))
ax= sns.heatmap(data1.corr(), cmap = 'RdBu_r', cbar=True, annot=True, linewidths=0.5, ax=ax)
plt.show()

data2=pd.DataFrame(index=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'],columns=["count","mean","std","min","25%","50%","75%","max"], data=Description)

f, ax = plt.subplots(figsize=(10,10))  
sns.heatmap(data2, annot=True, cmap="Blues", fmt='.0f', ax=ax, linewidths=2, cbar=False, annot_kws={"size":14}) 
plt.xticks(size=16) 
plt.yticks(size=10, rotation=45) 
plt.ylabel("Variables")
plt.title("Descriptive Statistics", size=14) 
plt.show()

data1.corr()
data1.corr()['Outcome'].sort_values(ascending=False)
data1['Outcome'].value_counts()

The outcomes of these graphs can be summarized as follows:

1. Pair Plot: The pair plot provides a visual overview of the relationships between different diagnostic measurements. It helps               in identifying potential correlations or patterns between different features.

2. Box Plots: The box plots provide a statistical summary of the distribution of each feature. They help in identifying the                   median, quartiles, and potential outliers for each feature.

3. Histograms: The histograms provide a graphical representation of the frequency distribution of each feature. They help in                    understanding the spread and skewness of each feature's values.

4. Heatmap of Correlation: The heatmap provides a visual representation of the correlation between different features. It helps                            in identifying highly correlated features which might impact the model's performance.

5. Heatmap of Descriptive Statistics: This heatmap provides a visual summary of the descriptive statistics of each feature. It                                         helps in understanding the central tendency and dispersion of each feature's values.

6. Countplot: The countplot reveals that the dataset is imbalanced. The number of patients who don’t have diabetes is more than                those who do. This imbalance could potentially affect the performance of the predictive model and might need to                  be addressed during the preprocessing stage.

7. Correlation with Outcome: The correlation heatmap shows a high correlation between the ‘Outcome’ and the features ‘Glucose’,                              ‘BMI’, ‘Age’, and ‘Insulin’. These features could be particularly important in predicting the                                     outcome and can be prioritized when accepting input from the user for the prediction model.

data1.groupby('Outcome').mean()

data1[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data1[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

imputer = SimpleImputer(strategy='mean')
data_filled = pd.DataFrame(imputer.fit_transform(data1))
data_filled.columns = data1.columns
data_filled.index = data1.index
data_filled
data_filled.columns
data_filled.index
X = data_filled.drop('Outcome', axis=1)
y = data_filled['Outcome']
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X.shape, X_train.shape, X_test.shape)
model = svm.SVC()
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# Set up the RandomizedSearchCV object 
random_search = RandomizedSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=10), n_iter=25, random_state=1)


# Fit the model to the training data and find the best parameters
random_search.fit(X_train, y_train)


# Make predictions on the test set using the best model
y_pred = random_search.best_estimator_.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
bagging = BaggingClassifier(random_search.best_estimator_, max_samples=1.0, max_features=1.0)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
print('Bagging Accuracy:', accuracy_score(y_test, y_pred_bagging))
print('Bagging Confusion Matrix:', confusion_matrix(y_test, y_pred_bagging))
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 42)
logreg.fit(X_train, y_train)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred_log_reg = log_reg.predict(X_test)

print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred_log_reg))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 24, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

X_axis = list(range(1, 31))
acc = pd.Series()
x = range(1,31)

best_k = 1
best_accuracy = 0

for i in list(range(1, 31)):
    knn_model = KNeighborsClassifier(n_neighbors = i) 
    knn_model.fit(X_train, y_train)
    prediction = knn_model.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    acc = acc.append(pd.Series(accuracy))  # Store the accuracy in the 'acc' series
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = i

plt.plot(X_axis, acc)
plt.xticks(x)
plt.title("Finding best value for n_neighbors")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

print('Highest accuracy achieved: ',acc.values.max())
print('Highest accuracy achieved with k =', best_k, ':', best_accuracy)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
ranfor.fit(X_train, y_train)

importances = pd.DataFrame({'feature':X.columns,'importance':np.round(ranfor.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.plot.bar()
plt.title("Feature Importances")
plt.show()

# Make predictions and calculate accuracy
Y_pred_ranfor = ranfor.predict(X_test)
accuracy_ranfor = accuracy_score(y_test, Y_pred_ranfor)
print("Random Forest: " + str(accuracy_ranfor * 100))
from sklearn.metrics import confusion_matrix
y_pred_knn = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred_knn)
cm
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_knn))
def predict_diabetes(model):
    pregnancies = float(input("Enter the number of pregnancies: "))
    glucose = float(input("Enter the glucose level: "))
    blood_pressure = float(input("Enter the blood pressure: "))
    skin_thickness = float(input("Enter the skin thickness: "))
    insulin = float(input("Enter the insulin level: "))
    bmi = float(input("Enter the BMI: "))
    diabetes_pedigree_function = float(input("Enter the Diabetes Pedigree Function: "))
    age = float(input("Enter the age: "))

    # Create a numpy array from the user input
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])

    # Preprocess input_data here
    # Rescale the data
    input_data = sc.transform(input_data.reshape(1, -1))

    # Use the model to make predictions
    prediction = model.predict(input_data)

    # Return the prediction
    return prediction

# Use the trained model to make a prediction
prediction = predict_diabetes(knn)

# Print the prediction
if prediction[0] == 1:
    print("The model predicts that the patient has diabetes.")
else:
    print("The model predicts that the patient does not have diabetes.")
