# Decision Tree classification (Node Flooding)

# Part A - Data Preparation

# Import the necessary libraries
import numpy as np
import pandas as pd
import os
import graphviz
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# Set the working directory (set to directory containing the dataset)
os.chdir('C:\\Users\Brad\Desktop\Briefcase\Personal\GeorgiaTechMasters\CS7641_MachineLearning\Homework\Homework1')

# Import the dataset and collect some high-level information on it
dataset = pd.read_csv('NodeFlooding.csv')
print ("Dataset Length = ", len(dataset))
print ("Dataset Shape = ", dataset.shape)
print (dataset.head())

# Break up the dataset into X and Y components
X = dataset.iloc[:, [19, 1, 2, 4, 6, 7, 16, 17, 18, 20]].values
Y = dataset.iloc[:, 21].values
print(X[:10, :])
print(Y[:10])

# Encode the categorical data
# Encode the Independent variables
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X_Headers = np.array(['NodeStatus_B', 'NodeStatus_NB', 'NodeStatus_PNB',
                      'UtilBandwRate', 'PacketDropRate', 'AvgDelay_perSec',
                      'PctLostByteRate', 'PacketRecRate', '10RunAvgDropRate',
                      '10RunAvgBandwUse', '10RunDelay', 'FloodStatus'])
print(X_Headers)
print(X[:10, :])

# Encode the dependent variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
Y_Results = np.array(['0=Block', '1=NB-No_Block', '2=NB-Wait', '3=No_Block'])
print(Y_Results)
print(Y[:10])

# Split the dataset into the Training set and Test set (25% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25,
                                                    random_state = 0)

# Part B - Run Decision Tree model without pruning

# Also ran this model with min_impurity_decrease = 0.025 & 0.05

# Determine Decision Tree classifier on the training set & cross-validate
# Classifier can split leaves to lowest level possible
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,
                                    min_impurity_decrease = 0,
                                    min_samples_split = 2)
scores = cross_val_score(classifier, X_train, y_train, cv=10)
print('Mean = ', np.mean(scores))
# Based on cross-validation, should predict with 99.9% accuracy

# Fit the classifier and display decision tree results in PDF file
classifier.fit(X_train, y_train)
dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names = X_Headers)
graph = graphviz.Source(dot_data)
graph.render('Nodes')
# PDF file saves to working directory
# 16 layers from root to lowest-level leaves

# Part C - Run Decision Tree model with pruning

# Also ran this model with min_impurity_decrease = 0.05 and
# min_samples_split = 7.5% and 10% of total records

# Prune the Decision Tree & cross-validate
# Leverage both "Minimum Sample Split" and "Minimum Impurity Decrease"
# Use different entries for these and measure accuracy and tree layers
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,
                                    min_impurity_decrease = 0.025,
                                    min_samples_split = 54)
scores = cross_val_score(classifier, X_train, y_train, cv=10)
print('Mean = ', np.mean(scores))
# 0.025 and 5% of data were used for Min Impurity Decrease & Min Sample Split
# These were determined to be optimal based on trial and error
# Based on cross-validation, should predict with 85.4% accuracy

# Fit the classifier and display decision tree results in PDF file
classifier.fit(X_train, y_train)
dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names = X_Headers)
graph = graphviz.Source(dot_data)
graph.render('Nodes1')
# PDF file saves to working directory
# 8 layers from root to lowest-level leaves

# Part D - Make predictions on the test data based on training model chosen

# Make predictions on the test data and calculate the Confusion Matrix
y_pred = classifier.predict(X_test)
print(Y_Results)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy = ', accuracy_score(y_test, y_pred))
# Based on the test data, pruned model predicts with 84.4% accuracy
# Classifies all "Block" and "No Block" correctly
# Misclassifies between some "NB-No Block" and "NB-Wait"

# Part E - Determine model performance over several training/test splits

# Measure model accuracy over a variety of test set sizes
train = []
test = []
splits = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
for split in splits:
    # Split the dataset into the Training set and Test set varying test set size
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1 - split,
                                                        random_state = 0)
    # Use classifier parameters that have been pre-determined
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,
                                        min_impurity_decrease = .025,
                                        min_samples_split = 86)
    classifier.fit(X_train, y_train)
    train.append(1 - classifier.score(X_train, y_train))
    test.append(1 - classifier.score(X_test, y_test))

# Graph results
plt.plot(splits, train, color='blue', label='Training Set')
plt.plot(splits, test, color='orange', label='Test Set')
plt.legend()
plt.xlabel('Training Size Pct')
plt.axes()
plt.ylabel('Error Rate')
plt.title('Error Rate vs Training Size')
plt.grid(True)
plt.show()
# Both Training & Test error rates degrade with lower training sizes
# Test error rate degrades faster than training error rate at lower training sizes
