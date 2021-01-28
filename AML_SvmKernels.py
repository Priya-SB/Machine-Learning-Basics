"""
Support Vector Machine (SVM)

Priya Bannur
PA33 | 1032170692
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.filterwarnings("ignore") 

# Importing the dataset
df=pd.read_csv('BreastCancer.csv')
print('\nFirst 5 observations: \n',df.head())

# Data Preprocessing
print('\nMissing values : ', df.isnull().sum().values.sum())
# Visualize distribution for outliers
plt.rcParams['figure.figsize']=(14,6)

for i , var in enumerate(list(df.columns.values)):
    plt.subplot(2,3,i+1)
    sns.distplot(df[var] , color='b' , kde=True)
    plt.grid()
    plt.tight_layout()

# Separating Dependent and Independent variables
X, y = df.iloc[:,[2,4]], df['Outcome'] #2D

# Splitting the dataset into the Train set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
print(X_train, X_test)


'''
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

#----------------------------------------------------------------------------------
# Training & Testing the SVM model using different Kernel Functions of SVC
print('\n----Linear Kernel----')
c1= SVC(kernel = 'linear', random_state = 0)
c1.fit(X_train, y_train)
y_pred = c1.predict(X_test)
print('Confusion Matrix:\n',confusion_matrix(y_test, y_pred))
a1=accuracy_score(y_test, y_pred)*100
print('Accuracy= ',a1)

print('\n----Polynomial Kernel----')
c2= SVC(kernel = 'poly', random_state = 0)
c2.fit(X_train, y_train)
y_pred = c2.predict(X_test)
print('Confusion Matrix:\n',confusion_matrix(y_test, y_pred))
a2=accuracy_score(y_test, y_pred)*100
print('Accuracy= ',a2)

print('\n----Radial Basis Func Kernel----')
c3= SVC(kernel = 'rbf', random_state = 0)  # default
c3.fit(X_train, y_train)
y_pred = c3.predict(X_test)
print('Confusion Matrix:\n',confusion_matrix(y_test, y_pred))
a3=accuracy_score(y_test, y_pred)*100
print('Accuracy= ',a3)

print('\n----Sigmoid Kernel----')
c4= SVC(kernel = 'sigmoid', random_state = 0)
c4.fit(X_train, y_train)
y_pred = c4.predict(X_test)
print('Confusion Matrix:\n',confusion_matrix(y_test, y_pred))
a4=accuracy_score(y_test, y_pred)*100
print('Accuracy= ',a4)

# Visualizing kernel comparison
plt.figure()
sns.barplot(x = ['Linear', 'Polynomial', 'RBF', 'Sigmoid'],
            y = [a1,a2,a3,a4])

#----------------------------------------------------------------------------------
# Visualising the Training set results and plot the decision boundary
plt.figure()
h = .02  # step size in the mesh
C = 1.0  # SVM regularization parameter

# create a mesh to plot in
x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['Linear Kernel SVC',
          'Polynomial Kernel SVC',
          'RBF Kernel SVC',
          'Sigmoid Kernel SVC']

for i, clf in enumerate((c1,c2,c3,c4)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.PRGn, alpha=0.7)

    # Plot also the points
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=plt.cm.PRGn)
    plt.xlabel('Perimeter')
    plt.ylabel('Smoothness')
    plt.ylim([0.04,0.16])
    plt.title(titles[i])
plt.show()

#----------------------------------------------------------------------------------
# Visualising the Testing set results and plot the decision boundary
plt.figure()
h = .02  # step size in the mesh
C = 1.0  # SVM regularization parameter

# create a mesh to plot in
x_min, x_max = X_test.iloc[:, 0].min() - 1, X_test.iloc[:, 0].max() + 1
y_min, y_max = X_test.iloc[:, 1].min() - 1, X_test.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['Linear Kernel SVC',
          'Polynomial Kernel SVC',
          'RBF Kernel SVC',
          'Sigmoid Kernel SVC']

for i, clf in enumerate((c1,c2,c3,c4)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.7)

    # Plot also the points
    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap=plt.cm.coolwarm)
    plt.xlabel('Perimeter')
    plt.ylabel('Smoothness')
    plt.ylim([0.04,0.16])
    plt.title(titles[i])
plt.show()