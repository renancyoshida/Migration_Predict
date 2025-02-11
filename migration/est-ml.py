import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings("ignore")

from paths import PATH_OUT, PATH_FIG, PATH_IN
PATH='D:'

df=pd.read_csv(f'{PATH}\\od-est-clean.csv')


df['lowpop_dist']=np.where(df['population91_y']<df['population91_y'].median(), 1, 0)*df['dist']

# === train/cv/test split ===
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV

y=df['movers']
x=df.drop(columns=['movers', 'muni_id_d', 'muni_id_o'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.6, random_state=1)

# === estimate basic models ===
from sklearn.linear_model import LogisticRegression
from xgboost import  XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
 
# which model?
def pred_model(train_x, train_y, test_x, test_y, model_name, metric, rng=47): 
    model=model_name(random_state=rng)
    model.fit(train_x, train_y)
    y_score=model.predict_proba(test_x)[:, 1]  
    score=metric(test_y, y_score)
    return score

models=[LogisticRegression, RandomForestClassifier, XGBClassifier, AdaBoostClassifier]
metric=average_precision_score

score_dict={}
for model in models:
    score_dict[f'{model}']=pred_model(x_train, y_train, x_test, y_test, model, metric, 47)
   
scores = pd.DataFrame(list(score_dict.items()), columns=['Model', 'Score'])
scores.sort_values(by='Score', ascending=False, inplace=True) # xgboost is the best
print(scores)

# undersampling majority class, multiply uniform from 1 to 250
# separate majority and minority classes
df=pd.concat([x_train, y_train], axis=1)
zeros=df[df['movers']==0]
ones=df[df['movers']==1]

def downsample_and_train(ones, zeros, x_test, y_test, model_class, metric, seed=47):

    k_values = np.linspace(1, 250, 10, dtype=int)  # 5 values between 1 and 250
    results = {}

    zero_indices = zeros.index.to_numpy()
    
    for k in k_values:
        n_samples = k * len(ones)
        sampled_indices = np.random.choice(zero_indices, size=n_samples, replace=False)
        samples = zeros.loc[sampled_indices]
        # Combine minority and downsampled majority class
        downsampled = pd.concat([ones, samples])
        x_train_down = downsampled.drop(columns='movers')
        y_train_down = downsampled['movers']
        results[k] = pred_model(x_train_down, y_train_down, x_test, y_test, model_class, metric, seed)
    
    return results

downsample_and_train(ones, zeros, x_test, y_test, AdaBoostClassifier, metric)

# hyperpar tuning

max_depth = np.arange(3, 7, 1)
learning_rate = np.linspace(0.1, 0.5, 5)
n_estimators = np.arange(50, 250, 5)

param_grid = {'max_depth': max_depth, 'learning_rate': learning_rate, 'n_estimators': n_estimators}

model = XGBClassifier(random_state=47)
grid_search = GridSearchCV(model, param_grid, scoring='average_precision', cv=5)
grid_search.fit(x_train, y_train)
grid_search.best_params_

best_model = grid_search.best_estimator_

pred_model(x_train, y_train, x_test, y_test, best_model, metric, 47)



# plot auc-pr
from sklearn.metrics import precision_recall_curve, auc

models=[LogisticRegression, AdaBoostClassifier, RandomForestClassifier, XGBClassifier]
    
plt.figure(figsize=(8, 6))

for model_class in models:
    model = model_class(random_state=47)
    model.fit(x_train, y_train)
    y_pred_probs = model.predict_proba(x_test)[:, 1]  # Get probability scores
    precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, label=f"{model_class} (AUC={pr_auc:.3f})")

# Plot formatting
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()

# feature importance
model = XGBClassifier(random_state=47)
model.fit(x_train, y_train)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(x_train.shape[1]), importances[indices])
plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation=90)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

