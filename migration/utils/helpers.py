
# HELPER FUNCTIONS
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def gen_histograms(df, cov_list):
    """
    Generate histograms of all covariates in the same plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the covariates.
    cov_list : list
        List of covariates to be plotted.
        
    """
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 20))
    for i, ax in enumerate(axs.flat):
        cov=cov_list[i]
        sns.histplot(df[cov], bins=20, ax=ax)
        ax.set_title(cov)
    return axs

def pred_model(train_x, train_y, test_x, test_y, model_name, metric, rng=47): 
    """
    Fit a model and predict on test set.
    
    Parameters
    ----------
    train_x : pd.DataFrame
        Training set features.
    train_y : pd.Series
        Training set target.
    test_x : pd.DataFrame
        Test set features.
    test_y : pd.Series
        Test set target.
    model_name : class
        Model class to be used.
    metric : function
        Metric to be used.
    rng : int
        Random state.
    """
        
    model=model_name(random_state=rng)
    model.fit(train_x, train_y)
    y_prob=model.predict_proba(test_x)[:, 1]  
    return y_prob

def plot_feature_importance(model, x_train, y_train):
    """
    Get feature importance from a model.
    
    Parameters
    ----------
    model : class
        Model class to be used.
    x_train : pd.DataFrame
        Training set features.
    y_train : pd.Series
        Training set target.
    """
    model.fit(x_train, y_train)
    importances = 100*model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(x_train.shape[1]), importances[indices])
    plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    
    return plt

def plot_pr_curve(models, x_train, y_train, x_test, y_test):
    
    """
    Plot precision-recall curve for a list of models.
    
    Parameters
    ----------
    models : list
        List of model classes.
    x_train : pd.DataFrame
        Training set features.
    y_train : pd.Series
        Training set target.
    x_test : pd.DataFrame
        Test set features.
    y_test : pd.Series
        Test set target.
    """
    best_model = models[-1]
    plt.figure(figsize=(8, 6))
    for model_class in models:
        if model_class == best_model:
            model = best_model
        else:
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
    return plt

def tune_xgboost(x_train, y_train):
    """
    Tune hyperparameters for XGBoost model.
    
    Parameters
    ----------
    x_train : pd.DataFrame
        Training set features.
    y_train : pd.Series
        Training set target.
    """
    
    max_depth = np.arange(3, 7, 2)
    learning_rate = np.linspace(0.1, 0.5, 3)
    n_estimators = [50, 150]
    param_grid = {'max_depth': max_depth, 'learning_rate': learning_rate, 'n_estimators': n_estimators}
    
    model = XGBClassifier(random_state=47)
    grid_search = GridSearchCV(model, param_grid, scoring='average_precision', cv=2)
    grid_search.fit(x_train, y_train)
    grid_search.best_params_
    best_model = grid_search.best_estimator_
    return best_model