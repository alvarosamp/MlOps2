import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
exp_id = mlflow.create_experiment("Loan Prediction")
with mlflow.start_run(run_name = 'DecisionTreeClassifier') as run:
    mlflow.set_tag("model", "DecisionTreeClassifier")

mlflow.end_run()
n_estimators = 10
criterion = 'gini'
mlflow.log_param('n_estimators', n_estimators)
mlflow.log_param('criterion', criterion)
mlflow.log_metric('accuracy', 0.95)
