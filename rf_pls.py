import optuna
from optuna.integration import OptunaSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
base_folder=r'C:\\Users\\user\\git\\github\\py2312_rf_pls\\'
file_name=base_folder+'regression_pls.csv'
model_name=base_folder+'rf_obtuna.pickle'
columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','Target']
target_column='Target'
df=pd.read_csv(file_name, encoding='utf-8', engine='python', usecols=columns)
features=[c for c in df.columns if c !=target_column]
train, test=train_test_split(df, test_size=0.2, random_state=115)
X_train=train[features]
y_train=train[target_column].values
X_test=test[features]
y_test=test[target_column].values
params = {'n_estimators': optuna.distributions.IntDistribution(1,1000,log=True),
          'max_depth': optuna.distributions.IntDistribution(1,100,log=True),
          'max_features': optuna.distributions.IntDistribution(2,7,log=False)}
model=RandomForestRegressor(random_state=444)
optuna_search=OptunaSearchCV(
    model,
    params,
    cv=10,
    n_jobs=-1,
    n_trials=30,
    verbose=2
)
optuna_search.fit(X_train, y_train)
y_pred=optuna_search.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse:.3f}')
best_estimator=optuna_search.best_estimator_
print(f'Best estimator: {best_estimator}')
study=optuna_search.study_
print(f'Number of finished trials: {len(study.trials)}')
print(f'Best trial:')
trial=study.best_trial
print(f' Value: {trial.value}')
print(f' Params: {trial.params}')
optuna.visualization.plot_optimization_history(study)