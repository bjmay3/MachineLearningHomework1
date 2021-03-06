# Artificial Neural Network classification (Node Flooding)

# Part A - Data Preparation

# Import the necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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

# Part B - Run ANN model on the training data

# Scale the X_train and X_test data for ease of ANN calculation
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Determine the ANN classifier on the training set & cross-validate
# 3 hidden layers each with varying number of nodes which are multiples
# of number of attributes (12, 24, 36, 48)
accuracy = []
nodes = [12, 24, 36, 48]
for node in nodes:
    classifier = MLPClassifier(hidden_layer_sizes = (node, node, node))
    classifier.fit(X_train_scaled, y_train)
    scores = cross_val_score(classifier, X_train_scaled, y_train, cv=4)
    accuracy.append(np.mean(scores))
plt.plot(nodes, accuracy)
plt.xlabel('Nodes per Layer')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Nodes per Layer')
plt.grid(True)
plt.show()
# Maximum accuracy varies but generally occurs at either 24 or 36 nodes per
# layer.  I chose 24 nodes per layer.


# Part C - Make predictions on the test data based on training model chosen

# Redo the ANN classifier on the training set & cross-validate
# 3 hidden layers each with 24 nodes per layer
classifier = MLPClassifier(hidden_layer_sizes = (24, 24, 24))
classifier.fit(X_train_scaled, y_train)
scores = cross_val_score(classifier, X_train_scaled, y_train, cv=4)
print('Mean = ', np.mean(scores))
# Based on cross-validation, should predict with around 75% accuracy

# Make predictions on the test data and calculate the Confusion Matrix
y_pred = classifier.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print(Y_Results)
print(cm)
print('Accuracy = ', accuracy_score(y_test, y_pred))
# Based on the test data, model predicts with aorund 80% accuracy
# Generally, "Block" and "No Block" are classified correctly with no or few errors.
# Some misclassification for the "NB-No Block" and "NB-Wait" categories.

# Part D - Determine model performance over several training/test splits

# Measure model accuracy over a variety of test set sizes
train = []
test = []
splits = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
for split in splits:
    # Split the dataset into the Training set and Test set varying test set size
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1 - split,
                                                        random_state = 0)
    # Scale the X_train and X_test data for ease of ANN calculation
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Use classifier parameters that have been pre-determined
    classifier = MLPClassifier(hidden_layer_sizes = (24, 24, 24))
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
plt.grid(b = True, which = 'both')
plt.show()
# Training error rate stays fairly consistent with test error rate.
# 35% training size case failed to converge after 200 iterations.
