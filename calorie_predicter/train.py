import pandas as pd
import mlflow
import mlflow.sklearn
import cupy as cp
from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

device_id=0
cp.cuda.Device(device_id).use()
seed = 42

kappa_scorer = make_scorer(cohen_kappa_score)

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

#X_train, y_train = cp.array(X_train.values), cp.array(y_train)
#X_test, y_test = cp.array(X_test), cp.array(y_test)


parameters = {
    'learning_rate': [0.1, 0.01, 0.001], 
    #'max_depth': [3, 5, 7],
    #'colsample_bytree': [0.6, 0.8, 1.0],
    #'n_estimators': [50, 100, 150]
}

xgb_clf = XGBClassifier(
                            objective='multi:softmax',
                            num_class=13,
                            random_state=seed,
                            #device = "cuda"
                       )

clf = GridSearchCV(xgb_clf, parameters, scoring='accuracy', n_jobs=-1)

clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)

best_params = clf.best_params_

training_accuracy = accuracy_score(y_train_pred, y_train)
testing_accuracy = accuracy_score(y_pred, y_test)
training_kappa = cohen_kappa_score(y_train_pred, y_train)
testing_kappa = cohen_kappa_score(y_pred, y_test)

metrics_dict = {
    'training_accuracy': training_accuracy,
    'testing_accuracy': testing_accuracy,
    'training_kappa': training_kappa,
    'testing_kappa': testing_kappa
}

for metric_name, metric_value in metrics_dict.items():
    print(f'{metric_name}: {metric_value}')

mlflow.set_experiment("training_experiment")
experiment = mlflow.get_experiment_by_name("training_experiment")

# Log parameters, metrics, and model artifacts with MLflow
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Log grid search parameters
    for key, value in parameters.items():
        mlflow.log_param(f'grid_search_{key}', value)
    
    # Log metrics
    for metric_name, metric_value in metrics_dict.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Log best parameters from grid search
    for key, value in clf.best_params_.items():
        mlflow.log_param(f'best_{key}', value)
    
    # Log XGBoost model
    mlflow.sklearn.log_model(clf.best_estimator_, 'xgboost_model')


