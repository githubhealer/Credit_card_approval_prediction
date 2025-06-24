import kagglehub
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

path = kagglehub.dataset_download("rikdifos/credit-card-approval-prediction")

print("Path to dataset files:", path)


for f,d,r in os.walk(path):
  print(f,d,r)
cred = pd.read_csv(f+'/credit_record.csv')
app = pd.read_csv(f+'/application_record.csv')

print("Printing first 5 columns of application_record.csv")
print(app.head())
print("Printing first 5 columns of credit_record.csv")
print(cred.head())

#Univariate analysis
sns.set(rc={'figure.figsize':(10,5)})
print(app['OCCUPATION_TYPE'].value_counts())
sns.countplot(x='OCCUPATION_TYPE',data=app,palette='Set2')
plt.show()
print("Types of house of people")
print(app['NAME_HOUSING_TYPE'].value_counts())
sns.set(rc={'figure.figsize':(10,5)})
sns.countplot(x='NAME_HOUSING_TYPE',data=app,palette='Set2')
plt.show()
print("Income types of a person")
print(app['NAME_INCOME_TYPE'].value_counts())
sns.set(rc={'figure.figsize':(10,5)})
sns.countplot(x='NAME_INCOME_TYPE',data=app,palette='Set2')
plt.show()

#Multivariate analysis
print("No of people working status")
print(app['OCCUPATION_TYPE'].value_counts())
fig,ax=plt.subplots(figsize=(10,5))
sns.heatmap(app.select_dtypes(include=np.number).corr(),annot=True)
plt.show()

#Descriptive analysis
print("Application Description")
print(app.describe())
print("-"*100)
print("Credit Card Description")
print(cred.describe())

print("Before removal of Duplicates",app.shape)
app = app.drop_duplicates(subset='ID',keep='first')
print("After removal of Duplicates",app.shape)

print("NULL Values:")
print(app.isnull().mean())

app['CNT_FAM_MEMBERS'] = app['CNT_FAM_MEMBERS'] + app['CNT_CHILDREN']
app = app.drop(['CNT_CHILDREN','FLAG_PHONE','FLAG_EMAIL','FLAG_WORK_PHONE','OCCUPATION_TYPE','FLAG_MOBIL'],axis=1)
print(app.columns)

app['DAYS_BIRTH'] = np.abs(app['DAYS_BIRTH']/365)
app['DAYS_EMPLOYED'] = app['DAYS_EMPLOYED']/365
print(app.head())

housing_type = {'House / apartment': 'House / apartment','With parents': 'With parents','Municipal apartment': 'House / apartment','Rented apartment': 'House / apartment','Office apartment': 'House / apartment','Co-op apartment': 'House / apartment'}

income_type = {'Commercial associate': 'Working','State servant': 'Working','Working': 'Working','Pensioner': 'Pensioner','Student': 'Student'}

education_type = {'Secondary / secondary special': 'secondary','Lower secondary': 'secondary','Higher education': 'Higher education','Incomplete higher': 'Higher education','Academic degree': 'Academic degree'}

family_status = {'Single / not married': 'Single','Separated': 'Single','Widow': 'Single','Civil marriage': 'Married','Married': 'Married'}

app['NAME_HOUSING_TYPE'] = app['NAME_HOUSING_TYPE'].map(housing_type)
app['NAME_INCOME_TYPE'] = app['NAME_INCOME_TYPE'].map(income_type)
app['NAME_EDUCATION_TYPE'] = app['NAME_EDUCATION_TYPE'].map(education_type)
app['NAME_FAMILY_STATUS'] = app['NAME_FAMILY_STATUS'].map(family_status)

print(app['NAME_EDUCATION_TYPE'].value_counts())

print("Information about credit_record.csv",cred.info())

grouped = cred.groupby('ID')
pivot_tb = cred.pivot(index='ID', columns='MONTHS_BALANCE', values='STATUS')
pivot_tb['open_month'] = grouped['MONTHS_BALANCE'].min()
pivot_tb['end_month'] = grouped['MONTHS_BALANCE'].max()
pivot_tb['window'] = pivot_tb['end_month'] - pivot_tb['open_month']
pivot_tb['window'] += 1
pivot_tb['paid_off'] = pivot_tb[pivot_tb.iloc[:, 0:61] == 'C'].count(axis=1)
pivot_tb['pastdue_1-29'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '0'].count(axis=1)
pivot_tb['pastdue_30-59'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '1'].count(axis=1)
pivot_tb['pastdue_60-89'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '2'].count(axis=1)
pivot_tb['pastdue_90-119'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '3'].count(axis=1)
pivot_tb['pastdue_120-149'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '4'].count(axis=1)
pivot_tb['pastdue_over_150'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '5'].count(axis=1)
pivot_tb['no_loan'] = pivot_tb[pivot_tb.iloc[:, 0:61] == 'X'].count(axis=1)
pivot_tb['ID'] = pivot_tb.index

def feature_engineering_target(data):
    good_or_bad = []
    for index, row in data.iterrows():
        paid_off = row['paid_off']
        over_1 = row['pastdue_1-29']
        over_30 = row['pastdue_30-59']
        over_60 = row['pastdue_60-89']
        over_90 = row['pastdue_90-119']
        over_120 = row['pastdue_120-149'] + row['pastdue_over_150']
        no_loan = row['no_loan']

        overall_pastdues = over_1 + over_30 + over_60 + over_90 + over_120

        if overall_pastdues == 0:
            if paid_off >= no_loan or paid_off <= no_loan:
                good_or_bad.append(1)
            elif paid_off == 0 and no_loan == 1:
                good_or_bad.append(1)

        elif overall_pastdues != 0:
            if paid_off > overall_pastdues:
                good_or_bad.append(1)
            elif paid_off <= overall_pastdues:
                good_or_bad.append(0)

        elif paid_off == 0 and no_loan != 0:
            if overall_pastdues <= no_loan or overall_pastdues >= no_loan:
                good_or_bad.append(0)

        else:
            good_or_bad.append(1)

    return good_or_bad

target = pd.DataFrame()
target['ID'] = pivot_tb.index
target['paid_off'] = pivot_tb['paid_off'].values
target['#_of_pastdues'] = (
    pivot_tb['pastdue_1-29'].values +
    pivot_tb['pastdue_30-59'].values +
    pivot_tb['pastdue_60-89'].values +
    pivot_tb['pastdue_90-119'].values +
    pivot_tb['pastdue_120-149'].values +
    pivot_tb['pastdue_over_150'].values
)
target['no_loan'] = pivot_tb['no_loan'].values
target['target'] = feature_engineering_target(pivot_tb)
print(target['target'].value_counts())
target

credit_app = app.merge(target,how='inner',on='ID')
credit_app.drop('ID',axis=1,inplace=True)
credit_app.to_csv('credit_app.csv',index=False)
print(credit_app.head())

cg = LabelEncoder()
oc = LabelEncoder()
own_r = LabelEncoder()
it = LabelEncoder()
et = LabelEncoder()
fs = LabelEncoder()
ht = LabelEncoder()
credit_app['CODE_GENDER'] = cg.fit_transform(credit_app['CODE_GENDER'])
credit_app['FLAG_OWN_CAR'] = oc.fit_transform(credit_app['FLAG_OWN_CAR'])
credit_app['FLAG_OWN_REALTY'] = own_r.fit_transform(credit_app['FLAG_OWN_REALTY'])
credit_app['NAME_INCOME_TYPE'] = it.fit_transform(credit_app['NAME_INCOME_TYPE'])
credit_app['NAME_EDUCATION_TYPE'] = et.fit_transform(credit_app['NAME_EDUCATION_TYPE'])
credit_app['NAME_FAMILY_STATUS'] = fs.fit_transform(credit_app['NAME_FAMILY_STATUS'])
credit_app['NAME_HOUSING_TYPE'] = ht.fit_transform(credit_app['NAME_HOUSING_TYPE'])
print(credit_app.head())

print("Mapping :", list(zip(cg.classes_, range(len(cg.classes_)))))
print("Mapping :", list(zip(oc.classes_, range(len(oc.classes_)))))
print("Mapping :", list(zip(own_r.classes_, range(len(own_r.classes_)))))
print("Mapping :", list(zip(it.classes_, range(len(it.classes_)))))
print("Mapping :", list(zip(et.classes_, range(len(et.classes_)))))
print("Mapping :", list(zip(fs.classes_, range(len(fs.classes_)))))
print("Mapping :", list(zip(ht.classes_, range(len(ht.classes_)))))

x = credit_app.drop('target',axis=1)
y = credit_app['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

def logistic_regression(x_train,x_test,y_train,y_test):
  print("Logistic Regression")
  lr=LogisticRegression(solver='lbfgs')
  lr.fit(x_train,y_train)
  y_pred=lr.predict(x_test)
  print("\n Confusion matrix:",confusion_matrix(y_test,y_pred))
  print("\n Classification report:",classification_report(y_test,y_pred))
  print("\n F1 score:",f1_score(y_test,y_pred))
  print("-"*100)
  return lr
logistic_regression(x_train,x_test,y_train,y_test)

def random_forest(x_train,x_test,y_train,y_test):
  print("Random Forest")
  rf = RandomForestClassifier()
  rf.fit(x_train, y_train)
  y_pred = rf.predict(x_test)
  print("\n Confusion matrix:",confusion_matrix(y_test,y_pred))
  print("\n Classification report:",classification_report(y_test,y_pred))
  print("\n F1 score:",f1_score(y_test,y_pred))
  print("-"*100)
  return rf
random_forest(x_train,x_test,y_train,y_test)

def xgboost(x_train,x_test,y_train,y_test):
  print("XGBoost")
  gb = GradientBoostingClassifier()
  gb.fit(x_train, y_train)
  y_pred = gb.predict(x_test)
  print("\n Confusion matrix:",confusion_matrix(y_test,y_pred))
  print("\n Classification report:",classification_report(y_test,y_pred))
  print("\n F1 score:",f1_score(y_test,y_pred))
  print("-"*100)
  return gb
xgboost(x_train,x_test,y_train,y_test)

def decision_tree_before_hypertuning(x_train,x_test,y_train,y_test):
  print("Decision Tree")
  dt=DecisionTreeClassifier()
  dt.fit(x_train,y_train)
  y_pred=dt.predict(x_test)
  print("\n Confusion matrix:",confusion_matrix(y_test,y_pred))
  print("\n Classification report:",classification_report(y_test,y_pred))
  print("\n F1 score:",f1_score(y_test,y_pred))
  print("-"*100)
  return dt
decision_tree_before_hypertuning(x_train,x_test,y_train,y_test)

def decision_tree_after_hypertuning(x_train, x_test, y_train, y_test):
    print("Decision Tree - Starting with GridSearchCV")

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )

    print("Fitting the model with parameter grid...")
    grid_search.fit(x_train, y_train)

    print("Best parameters found:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)

    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nF1 score:", f1_score(y_test, y_pred))
    print("-" * 100)

    return best_model
decision_tree_after_hypertuning(x_train, x_test, y_train, y_test)

def compare_models(x_train,x_test,y_train,y_test):
  lr = logistic_regression(x_train,x_test,y_train,y_test)
  rf = random_forest(x_train,x_test,y_train,y_test)
  gb = xgboost(x_train,x_test,y_train,y_test)
  dt = decision_tree_before_hypertuning(x_train,x_test,y_train,y_test)
  dt2 = decision_tree_after_hypertuning(x_train,x_test,y_train,y_test)
  return lr,rf,gb,dt,dt2
compare_models(x_train,x_test,y_train,y_test)

#Save the best model
model=decision_tree_after_hypertuning(x_train,x_test,y_train,y_test)
filename = 'c_card_model.pkl'
pickle.dump(model, open(filename, 'wb'))