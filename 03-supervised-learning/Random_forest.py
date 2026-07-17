import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('heart.csv')
print(df.shape)

x = df.iloc[:,0:-1] #iloc 0 to -1 means all rows and all columns except last one
y = df.iloc[:,-1] #iloc -1 means all rows and last column

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


"""lets do a comparitive study"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

models = {
    'Random Forest': RandomForestClassifier(max_depth=8, max_features=0.2, max_samples=0.75, n_estimators=20, random_state=42),  
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}
results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"{model_name} - Accuracy: {accuracy:.4f}")
    
    
    cross_val_scores = np.mean(cross_val_score(model, X_train, y_train, cv=5))
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Cross-Validation Score: {cross_val_scores:.4f}")
    
"""we will use grid search to find the best hyperparameters for our random forest model"""

from sklearn.model_selection import GridSearchCV
# Number of trees in random forest
n_estimators = [20,60,100,120]

# Number of features to consider at every split
max_features = [0.2,0.6,1.0]

# Maximum number of levels in tree
max_depth = [2,8,None]

# Number of samples
max_samples = [0.5,0.75,1.0]

# 108 diff random forest train
param_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'max_samples': max_samples
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

"""we will use optuna now"""

import optuna
def objective(trial):
    n_estimators = trial.suggest_categorical('n_estimators', [20, 60, 100, 120])
    max_features = trial.suggest_categorical('max_features', [0.2, 0.6, 1.0])
    max_depth = trial.suggest_categorical('max_depth', [2, 8, None])
    max_samples = trial.suggest_categorical('max_samples', [0.5, 0.75, 1.0])
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        max_samples=max_samples,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=5)) 
    
    return cv_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=-1, show_progress_bar=False)
print("Best Hyperparameters:", study.best_params)
print("Best Cross-Validation Score:", study.best_value)
