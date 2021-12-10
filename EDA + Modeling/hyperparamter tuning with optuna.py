#xgb performs better among all model so we will choose it
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#hyperparameter tuning for xgboost
import optuna
def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1),
        'max_depth': trial.suggest_int('max_depth', 2, 9),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 1),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.5, 1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'objective': 'binary:logistic',
    }
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return -roc_auc_score(y_test, pred)

#run hyperparameter tuning
study = optuna.create_study()
study.optimize(objective, n_trials=10)

#show results
print(study.best_params)
print(study.best_value)

O/P-
{'learning_rate': 0.03125999577306311, 'max_depth': 9, 'min_child_weight': 5.145059115450818e-06, 'gamma': 0.02810715905283532, 'subsample': 0.8238728569946043,
 'colsample_bytree': 0.8049634980552405, 'colsample_bylevel': 0.9449990388848694, 'reg_alpha': 0.022932992629365996, 'reg_lambda': 2.5406456895163724e-06, 'n_estimators': 199}
-0.8993331723408908