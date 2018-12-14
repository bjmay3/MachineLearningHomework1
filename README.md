All coding done in Python.

Ten (10) separate strings of code exist.  These are as follows:
	a. Decision Tree - Car Evaluation dataset
	b. Decision Tree - Node Flooding dataset
	c. Boosting - Car Evaluation dataset
	d. Boosting - Node Flooding dataset
	e. KNN - Car Evaluation dataset
	f. KNN - Node Flooding dataset
	g. SVM - Car Evaluation dataset
	h. SVM - Node Flooding dataset
	i. ANN - Car Evaluation dataset
	j. ANN - Node Flooding dataset

Each string of code, except the two (2) Decision Tree strings, is broken into four (4) parts.
	a. Part A - Data Preparation:  loads libraries, loads data, manipulates data as necessary, establishes training and test sets.
	b. Part B - Run the model on the training data:  runs each model on the training set and provides cross-validated results.
	c. Part C - Run the model on the test data: runs each model on the test set, obtains confusion matrix of results, measures model accuracy.
	d. Part D - Model performance over different training/test splits: uses different training sizes and evaluates model performance at each.

For the two (2) Decision Tree code strings, there are five (5) parts to the code.
	a. Part A - Data Preparation:  loads libraries, loads data, manipulates data as necessary, establishes training and test sets.
	b. Part B - Run the unpruned model on the training data:  runs the model without pruning on the training set and provide cross-validated results.
	c. Part C - Run the pruned model on the training data:  runs the model with pruning on the training set and provide cross-validated results.
	d. Part D - Run the model on the test data: runs the model on the test set, obtains confusion matrix of results, measures model accuracy.
	e. Part E - Model performance over different training/test splits: uses different training sizes and evaluates model performance at each.

At a minimum, run each part of the code separately.	The following results will be displayed by part:
	a. Data Preparation - various data sizes and data displays to indicate that download, transformation, and breaking into X and y components happened correctly.
	b. Model run on training data - average of cross-validated results, sometimes graphs of accuracy vs. certain attributes (# of neighbors, kernel type, etc.).
	c. Model run on test data - confusion matrix, accuracy based on confusion matrix
	d. Model performance - graph of Error Rate vs. Training size for both training data and test data

Attribution:  Some code was "borrowed" from elsewhere.  The following gives credit to the places where various pieces of code were obtained.
	a. Udemy Super Data Science course - Data Preparation, predictions from test set, confusion matrix development, Decision Tree, KNN, SVM model training.
	b. AdaBoost code - Boosting model training using AdaBoost.
	c. ANN code - ANN model training using MLPClassifier.

Attribution Websites
	a. Udemy Super Data Science course - https://www.udemy.com/datascience
	b. AdaBoost code - https://www.youtube.com/watch?v=X3Wbfb4M33w
			 - scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
	c. ANN code - https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html
		    - scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
