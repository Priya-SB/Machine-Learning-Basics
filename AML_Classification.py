"""
Priya Bannur
PA 33, 1032170692
------------------
AML Lab 3 : Calssification Algorithms
"""

# Import Libraries and Modules
import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_curve, roc_auc_score
SEED = 42  
import warnings
warnings.filterwarnings("ignore")

# Read the .csv dataset
df = pd.read_csv('BreastCancer.csv')

# Printing Overview of the dataset
print('\n-----------------------Data Overview-------------------------\n')
print('\nRows : ', df.shape[0])
print('\nColumns : ', df.shape[1])
print('\nFeatures :\n ', df.loc[[0]])
print('\nMissing values : ', df.isnull().sum().values.sum())
print('\nUnique values :\n ', df.nunique())
print('\nFirst 5 observations: \n',df.head())
print('\nDescrption:\n',df.info(),'\n', df.describe())

# Checking balance of Target column
print('\nTarget Balance :\n ',df.Outcome.value_counts())
plt.figure()
sns.set(font_scale=0.8)
df['Outcome'].value_counts().plot(kind='bar', figsize=(5, 3), rot=0)
plt.xlabel("Diagnosis")
plt.ylabel("Count of People")
plt.title('Target Balance')

# Proportion of positive vs negative cancer
print('\nBalance Ratio : ',df.Outcome.value_counts()[1] / df.Outcome.count())


#------------------------------------------------------------------------------
# Plots
# Pair Plot to visualize relationship between variables
plt.figure()
sns.pairplot(df, hue='Outcome', plot_kws=dict(alpha=0.5, edgecolor='none'), 
             height=2, aspect=1.1)


# Correlation Matrix Heatmap Visualization
plt.figure()
sns.set(style="white")
# Generate a mask for the upper triangle
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure to control size of heatmap
fig, ax = plt.subplots(figsize=(8,8))
# Create a custom color palette
cmap = sns.diverging_palette(255, 10, as_cmap=True)  
# Plot the heatmap
sns.heatmap(df.corr(), mask=mask, annot=True, square=True, cmap=cmap , 
            vmin=-1, vmax=1, ax=ax)  # annot display corr label
# Prevent Heatmap Cut-Off Issue
bottom, top = ax.get_ylim()
ax.set_ylim(bottom, top)


# To analyse feature-outcome distribution in visualisation
features = ['radius','texture','perimeter','area','smoothness']
ROWS, COLS = 3,2
fig, ax = plt.subplots(ROWS, COLS, figsize=(18,8) )
row, col = 0, 0
for i, feature in enumerate(features):
    if col == COLS - 1:
        row += 1
    col = i % COLS
    
    df[df.Outcome==0][feature].hist(bins=35, color='green', 
                                    alpha=0.5, ax=ax[row, col]).set_title(feature)
    df[df.Outcome==1][feature].hist(bins=35, color='red', 
                                    alpha=0.7, ax=ax[row, col])
  
plt.legend(['No Cancer', 'Cancer'])
fig.subplots_adjust(hspace=1)

#Create feature X and target y dataset
X, y = df.drop('Outcome', axis=1), df['Outcome']
print('Dataset Shape',X.shape, y.shape)

# To look for top features using Random Forest
# Create decision tree classifer object
rfc = RandomForestClassifier(random_state=SEED, n_estimators=100)

# Train model, note that NO scaling is required
rfc_model = rfc.fit(X, y)

# Plot the top features based on its importance
(pd.Series(rfc_model.feature_importances_, index=X.columns)
    .nlargest(5)   # can adjust based on how many top features you want
    .plot(kind='barh', figsize=[8,4])
    .invert_yaxis()) # Descending order
plt.yticks(size=8)
plt.title('Top Features by RF') 

#------------------------------------------------------------------------------

# For linear data and model, p-value < 0.05 indicates a significant feature
X = sm.add_constant(X)  # need to add this to define the Intercept

# model / fit / summarize results
model = sm.OLS(y, X)
result = model.fit()
print('\nSummary using StatsModel\n',result.summary())

X = df.drop('Outcome', axis=1)   # axis=0 for row, axis=1 for column
y = df['Outcome']

# Split data to 80:20 ratio for train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, 

                                                    random_state=SEED, stratify=y)
print('\n-----------------------Modelling-------------------------\n')
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test: ', X_test.shape)
print('y_test: ', y_test.shape)

# Cross Validation
kf1 = KFold(n_splits=5, shuffle=True, random_state=SEED)   
# this may result in imbalance classes in each fold

# Stratified KFold shuffles after splitting the data, less imbalance
kf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# to give model baseline report in dataframe 
def baseline_report_1(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    accuracy =np.mean(cross_val_score(model, X_train, y_train, cv=kf1, scoring='accuracy'))
    precision=np.mean(cross_val_score(model, X_train, y_train, cv=kf1, scoring='precision'))
    recall   =np.mean(cross_val_score(model, X_train, y_train, cv=kf1, scoring='recall'))
    f1score  =np.mean(cross_val_score(model, X_train, y_train, cv=kf1, scoring='f1'))
    rocauc   =np.mean(cross_val_score(model, X_train, y_train, cv=kf1, scoring='roc_auc'))
    y_pred   =model.predict(X_test)
    logloss  =log_loss(y_test, y_pred)   # SVC & LinearSVC unable to use cvs

    df_model1 = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'rocauc'       : [rocauc],
                             'logloss'      : [logloss]
                                  })   
    return df_model1


def baseline_report_2(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    accuracy =np.mean(cross_val_score(model, X_train, y_train, cv=kf2, scoring='accuracy'))
    precision=np.mean(cross_val_score(model, X_train, y_train, cv=kf2, scoring='precision'))
    recall   =np.mean(cross_val_score(model, X_train, y_train, cv=kf2, scoring='recall'))
    f1score  =np.mean(cross_val_score(model, X_train, y_train, cv=kf2, scoring='f1'))
    rocauc   =np.mean(cross_val_score(model, X_train, y_train, cv=kf2, scoring='roc_auc'))
    y_pred   =model.predict(X_test)
    logloss  =log_loss(y_test, y_pred)   # SVC & LinearSVC unable to use cvs

    df_model2 = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'rocauc'       : [rocauc],
                             'logloss'      : [logloss]
                                  })   
    return df_model2

# To evaluate baseline models
gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()
logit = LogisticRegression()
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
svc = SVC()
linearsvc = LinearSVC()

# To concat all models
df_models1 = pd.concat([
    baseline_report_1(randomforest, X_train, X_test, y_train, y_test, 'RandomForest'),
    baseline_report_1(decisiontree, X_train, X_test, y_train, y_test, 'DecisionTree'),
    baseline_report_1(logit, X_train, X_test, y_train, y_test, 'LogisticRegression'),
    baseline_report_1(gnb, X_train, X_test, y_train, y_test, 'GaussianNB'),
    baseline_report_1(svc, X_train, X_test, y_train, y_test, 'SVC'),
    baseline_report_1(knn, X_train, X_test, y_train, y_test, 'KNN'),
    baseline_report_1(mnb, X_train, X_test, y_train, y_test, 'MultinomialNB'),
    baseline_report_1(linearsvc, X_train, X_test, y_train, y_test, 'LinearSVC'),
    baseline_report_1(bnb, X_train, X_test, y_train, y_test, 'BernoulliNB')
                       ], axis=0).reset_index()
df_models1 = df_models1.drop('index', axis=1)
print('\nKFold\n', df_models1)

df_models2 = pd.concat([
    baseline_report_2(randomforest, X_train, X_test, y_train, y_test, 'RandomForest'),
    baseline_report_2(decisiontree, X_train, X_test, y_train, y_test, 'DecisionTree'),
    baseline_report_2(logit, X_train, X_test, y_train, y_test, 'LogisticRegression'),
    baseline_report_2(gnb, X_train, X_test, y_train, y_test, 'GaussianNB'),
    baseline_report_2(svc, X_train, X_test, y_train, y_test, 'SVC'),
    baseline_report_2(knn, X_train, X_test, y_train, y_test, 'KNN'),
    baseline_report_2(mnb, X_train, X_test, y_train, y_test, 'MultinomialNB'),
    baseline_report_2(linearsvc, X_train, X_test, y_train, y_test, 'LinearSVC'),
    baseline_report_2(bnb, X_train, X_test, y_train, y_test, 'BernoulliNB')
                       ], axis=0).reset_index()
df_models2 = df_models2.drop('index', axis=1)
print('\nStratifieldKFold\n',df_models2)

#------------------------------------------------------------------------------
# Training and Testing classification models

print('\nTest Accuracies of Classification Models:')
randomforest.fit(X_train,y_train)
y_pred = randomforest.predict(X_test)
rf=accuracy_score(y_test, y_pred)*100
print('\nRandom Forest Test Accuracy: {0:.2f}'.format(rf))

decisiontree.fit(X_train,y_train)
y_pred = decisiontree.predict(X_test)
dt=accuracy_score(y_test, y_pred)*100
print('Decision Tree Test Accuracy: {0:.2f}'.format(dt))

logit.fit(X_train,y_train)
y_pred = logit.predict(X_test)
lr=accuracy_score(y_test, y_pred)*100
print('Logistic Regression Test Accuracy: {0:.2f}'.format(lr))

gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)
gn=accuracy_score(y_test, y_pred)*100
print('Gaussian Naive Bayes Test Accuracy: {0:.2f}'.format(gn))

svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
sv=accuracy_score(y_test, y_pred)*100
print('Support Vector Test Accuracy: {0:.2f}'.format(sv))

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
kn=accuracy_score(y_test, y_pred)*100
print('K Nearest Neighbours Test Accuracy: {0:.2f}'.format(kn))

mnb.fit(X_train,y_train)
y_pred = mnb.predict(X_test)
mn=accuracy_score(y_test, y_pred)*100
print('Multinomial Naive Bayes Test Accuracy: {0:.2f}'.format(mn))

linearsvc.fit(X_train,y_train)
y_pred = linearsvc.predict(X_test)
lsv=accuracy_score(y_test, y_pred)*100
print('Linear Support Vector Test Accuracy: {0:.2f}'.format(lsv))

bnb.fit(X_train,y_train)
y_pred = bnb.predict(X_test)
bn=accuracy_score(y_test, y_pred)*100
print('Bernoulli Naive Bayes Test Accuracy: {0:.2f}'.format(bn))

# Comparing test accuracies
plt.figure()
sns.lineplot(x = ['Gaussian NB', 'Multinomial NB', 'Bernoulli NB', 
                  'Support Vector', 'Linear SVC', 'Decision Tree', 
                  'Random Forest', 'K Nearest N', 'Logistic Regr' ], 
            y = [gn, mn, bn, sv, lsv, dt, rf, kn, lr ],
            marker='o')
plt.xticks(rotation=30)

#------------------------------------------------------------------------------

# ROC AUC Curve
plt.figure()
# predict probabilities
lr_probs = logit.predict_proba(X_test)
nb_probs = gnb.predict_proba(X_test)

# Keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
nb_probs = nb_probs[:, 1]
ns_probs = [0 for _ in range(len(y_test))]

# Calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
nb_auc = roc_auc_score(y_test, nb_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

# Summarize scores
print('\nNo Skill: ROC AUC=%.3f' % (ns_auc))
print('\nLogistic: ROC AUC=%.3f' % (lr_auc))
print('\nNaive Bayes: ROC AUC=%.3f' % (nb_auc))

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)

# Plot the roc curve for the models
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC curve')
plt.show()

#------------------------------------------------------------------------------