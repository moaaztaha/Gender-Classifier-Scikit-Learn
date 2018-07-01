from sklearn.tree import DecisionTreeClassifier #Import decision tree classifier
from sklearn.neighbors import KNeighborsClassifier #Import nearest neighbors algorithm 
from sklearn.neural_network import MLPClassifier   #Import multi-layer perception algorithm 
from numpy import argmax #Gets the index of the largest value in list
from sklearn.metrics import accuracy_score #Import liberary for measuring accuracy

#Data
#[Height, Weight, ShoeSize]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#The corresponding answer
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#Testing data
test = [[190,70,34]]

#Decision Tree Classifier
dt_clf = DecisionTreeClassifier()

#Fit the data using Decision Tree Classifier
dt_clf.fit(X,Y)

###############################################################################################################
#Nearest Neighbors Classifier
nn_clf = KNeighborsClassifier()

#Fit the data using Nearest Neighbour Classifier
nn_clf.fit(X,Y)

###############################################################################################################
#Multi-layer perception Algorithm Classifier
mlp_clf = MLPClassifier()

#Fit the data using Multi-layer perception Algorithm Classifier
mlp_clf.fit(X,Y)

#############################################################################################
#Making the predictions

#Decision Tree Prediction 
dt_prediction = dt_clf.predict(test)

#Nearest Neighbors Prediction
nn_prediction = nn_clf.predict(test)

#Multi-Layer Perception Prediction
mlp_prediction = mlp_clf.predict(test)

#############################################################################################
#Printing the predictions 
print("Predictions:")
print("Decision Tree:",dt_prediction)
print("Nearest Neighbors:",nn_prediction)
print("Multi-Layer Perception:",mlp_prediction)

#############################################################################################
#Comparing Accuracy 

#Decision Tree predicted Ys
dt_pred_y = dt_clf.predict(X)

#Calculating Accuracy 
dt_acc = accuracy_score(dt_pred_y,Y)

#Nearest Neighbors predicted Ys
nn_pred_y = nn_clf.predict(X)

#Calculating Accuracy
nn_acc = accuracy_score(nn_pred_y,Y)

#Multi-Layer Perception predicted Ys
mlp_pred_y = mlp_clf.predict(X)

#Calculating Accuracy
mlp_acc = accuracy_score(mlp_pred_y,Y)

#Index of Heighest Accuracy
index = argmax([dt_acc, nn_acc, mlp_acc])

#Dectionary of Classifiers
classifiers = {0:'Decision Tree', 1:'Nearest Neighbors', 2:'Multi-Layer Perception'}

#############################################################################################
#Printing Accuracy Scores and Comparison Results
print("***********************************************")
print("Decision Tree Accuracy:",dt_acc)
print("Nearest Neighbors Accuracy:",nn_acc)
print("Multi-Layer Perception Accuracy:",mlp_acc)
print("***********************************************")
print("The Most Accurate Classifier is: ", classifiers[index])