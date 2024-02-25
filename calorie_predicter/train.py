import mlflow
import mlflow.sklearn
from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

seed = 42

kappa_scorer = make_scorer(cohen_kappa_score)

parameters = {
    'learning_rate': [0.1, 0.01, 0.001], 
    #'max_depth': [3, 5, 7],
    #'colsample_bytree': [0.6, 0.8, 1.0],
    #'n_estimators': [50, 100, 150]
}

xgb_clf = XGBClassifier(objective='multi:softmax', 
                             num_class=13, 
                             random_state=seed, 
                             device = "cuda")

clf = GridSearchCV(xgb_clf, parameters, scoring='accuracy', n_jobs=-1)

clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train_tfidf)
y_pred = clf.predict(X_test_tfidf)

best_params = clf.best_params_

training_accuracy = accuracy_score(y_train_pred, y_train)
testing_accuracy = accuracy_score(y_pred, y_test)
training_kappa = cohen_kappa_score(y_train_pred, y_train)
testing_kappa = cohen_kappa_score(y_pred, y_test)
print('Accuracy training set score for xgboost after GridSearch:', training_accuracy)
print('Accuracy testing set score for xgboost after GridSearch:', testing_accuracy)
print('Kappa training set score for xgboost after GridSearch:', training_kappa)
print('Kappa testing set score for xgboost after GridSearch:', testing_kappa)

# Log parameters, metrics, and model artifacts with MLflow
with mlflow.start_run():
    mlflow.log_params(parameters)
    mlflow.log_metric('training_accuracy', training_accuracy)
    mlflow.log_metric('testing_accuracy', testing_accuracy)
    mlflow.log_metric('training_kappa', training_kappa)
    mlflow.log_metric('testing_kappa', testing_kappa)
    mlflow.log_params(clf.best_params_)
    mlflow.sklearn.log_model(clf.best_estimator_, 'xgboost_model')


