import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import  XGBClassifier
 
from helpers import tune_xgboost, plot_feature_importance, plot_pr_curve 

PATH='D:'
PATH_FIG=r'C:\Users\renan\Documents\Migration_Model\migration\figures'
df=pd.read_csv(f'{PATH}\\od-est-clean.csv')

# === train/cv/test split ===

y=df['movers']
x=df.drop(columns=['movers', 'muni_id_d', 'muni_id_o'])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.6, random_state=1)

# hyperparameter tuning
best_model = tune_xgboost(x_train, y_train)

# plot auc-pr
models=[DecisionTreeClassifier, XGBClassifier, best_model]
plot_pr_curve(models, x_train, y_train, x_test, y_test)
plt.savefig(f'{PATH_FIG}\\pr_curve.png')
plt.show()

# feature importance
model = XGBClassifier(random_state=47)
plot_feature_importance(model, x_train, y_train)
plt.savefig(f'{PATH_FIG}\\feature_importance.png')
plt.show()

