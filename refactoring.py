import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import preprocessing, metrics, tree #Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import GaussianNB
from sklearn.externals.six import StringIO
from IPython.display import Image  
import pydotplus
import csv

col_names = ['smell', 'refactoring', 'label']
# load dataset
pima = pd.read_csv("Datasets/CodeSmells.csv", header=0, names=col_names)

pima.head()

le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
smell_encoded=le.fit_transform(pima.smell)
refactoring_encoded=le.fit_transform(pima.refactoring)
label=le.fit_transform(pima.label)

with open('Smells.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Code Smell', 'Refactoring', 'Label'])
    i = 0
    while i < len(smell_encoded):
        filewriter.writerow([smell_encoded[i], refactoring_encoded[i], label[i]])
        i+=1
        
pima = pd.read_csv("Smells.csv", header=0, names=col_names)
pima.head()        

#split dataset in features and target variable
feature_cols = ['smell', 'refactoring']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=100)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy Decision Tree: ",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
                
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('smells.png')
Image(graph.create_png())

features=zip(smell_encoded,refactoring_encoded)

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features,label)

X = features # Features
y = label # Target variable

#Predict Output
expected = label
predicted= model.predict(features)

print("Predicted Value: ", predicted)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print("Accuracy Naive Bayes: ",metrics.accuracy_score(expected, predicted))