import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import mlflow
import os
import joblib
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")

#Load the dataset
def load_data(data_path):
    dataset = pd.read_csv(data_path)
    numerical_cols = dataset.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = dataset.select_dtypes(include='object').columns.tolist()
    for col in ['Loan_Status', 'Loan_ID']:
        if col in categorical_cols:
            categorical_cols.remove(col)
    return dataset, numerical_cols, categorical_cols

#Data Processing 
def preprocess_data(dataset, numerical_cols, categorical_cols):
    # Handling Missing Values
    for col in categorical_cols:
        dataset[col].fillna(dataset[col].mode()[0], inplace = True) # mode vai retornar o valor mais comum em uma coluna ou serie do pandas
    for col in numerical_cols:
        dataset[col].fillna(dataset[col].median(), inplace = True) # median vai retornar o valor mediano em uma coluna ou serie do pandas
    # Outlier Detection and Removal
    dataset[numerical_cols] = dataset[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))#x.clip(min, max) limita todos os valores de x para que fiquem entre min (5%) e max (95%). Valores abaixo de 5% viram 5%, e acima de 95% viram 95%.
    #Feature engineering
    """
    Como escolher a normalização/transformação ideal:

    1. Analise a distribuição da variável (use histograma ou describe()):
       - Se for aproximadamente normal: padronização (Z-score) é suficiente.
       - Se for muito assimétrica à direita (skewed): tente transformação logarítmica.
       - Se houver muitos outliers: prefira Robust Scaler.
       - Se precisa de valores entre 0 e 1 (ex: redes neurais): use Min-Max.
       - Se a variável tem valores negativos: use Yeo-Johnson (não log).

    2. Considere o algoritmo:
       - Modelos lineares, SVM, KNN: sensíveis à escala, prefira padronização ou min-max.
       - Árvores e florestas: menos sensíveis, normalização não é obrigatória.

    3. Teste e compare:
       - Sempre visualize antes/depois e avalie o impacto no desempenho do modelo.

    Exemplo prático:
    plt.hist(dataset['LoanAmount']); plt.show()
    Se for muito skewed, aplique log. Se já for normal, padronize.


    Usa mediana e intervalo interquartil (IQR) para reduzir o impacto de outliers.
    Quando usar: Quando há muitos outliers nos dados.
    """
    dataset['LoanAmount'] = np.log(dataset['LoanAmount']).copy()
    dataset['Total_Income'] = np.log(dataset['ApplicantIncome'] + dataset['CoapplicantIncome']).copy()
    #Dropping applicant income and coapplicant income
    dataset.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1, inplace=True)
    #Encoding Categorical Variables
    le = LabelEncoder()
    for col in categorical_cols:
        dataset[col] = le.fit_transform(dataset[col])

    #Encode the target columns
    dataset['Loan_Status'] = le.fit_transform(dataset['Loan_Status'])
    X = dataset.drop(['Loan_ID', 'Loan_Status'], axis = 1)
    y = dataset['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test

def random_forest(random_state = 42):
    rf = RandomForestClassifier(random_state= random_state)
    #Aplicando o grid search
    param_grid_forest = {
        'n_estimators': [200,400, 700],
        'max_depth': [10,20,30],
        'criterion' : ["gini", "entropy"],
        'max_leaf_nodes': [50, 100]
    }
    grid_forest = GridSearchCV(estimator = rf, param_grid = param_grid_forest, cv = 5, n_jobs = -1, verbose = 0)
    model_forest = grid_forest.fit(X_train, y_train)
    # Salva o melhor modelo no MLflow
    mlflow.sklearn.log_model(model_forest.best_estimator_, "random_forest_model")
    # Salva o modelo localmente usando joblib
    joblib.dump(model_forest.best_estimator_, r"C:\Users\alvaro.careli\Documents\DeployMLModels\Mlops\Bootcamp_MlOps\MlOps2\MlFlow\Modelos\random_forest_model.joblib")
    return model_forest

def logistic_regression(random_state = 42):
    log_reg = LogisticRegression(random_state = random_state)
    param_grid_log = {
        'C': [100, 10, 1.0, 0.1, 0.01],
        'penalty': ['l1','l2'],
        'solver':['liblinear']
    }
    grid_logistic = GridSearchCV(estimator = log_reg, param_grid = param_grid_log, cv = 5, n_jobs = -1, verbose = 0)
    model_logistic = grid_logistic.fit(X_train, y_train)
    mlflow.sklearn.log_model(model_logistic.best_estimator_, "logistic_regression_model")
    joblib.dump(model_logistic.best_estimator_, r"C:\Users\alvaro.careli\Documents\DeployMLModels\Mlops\Bootcamp_MlOps\MlOps2\MlFlow\Modelos\logistic_regression_model.joblib")
    return model_logistic

def decision_tree(random_state = 42):
    dec_tree= DecisionTreeClassifier(random_state = random_state)
    param_grid_tree = {
    "max_depth": [3, 5, 7, 9, 11, 13],
    'criterion' : ["gini", "entropy"],
}
    model_tree = GridSearchCV(estimator = dec_tree, param_grid = param_grid_tree, cv = 5, n_jobs = -1, verbose = 0)
    model_tree = model_tree.fit(X_train, y_train)
    mlflow.sklearn.log_model(model_tree.best_estimator_, "decision_tree_model")
    joblib.dump(model_tree.best_estimator_, r"C:\Users\alvaro.careli\Documents\DeployMLModels\Mlops\Bootcamp_MlOps\MlOps2\MlFlow\Modelos\decision_tree_model.joblib")
    return model_tree

#Eval function
def eval_metrics(actual, pred, pred_proba=None):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    # Use probabilidades para ROC/AUC se disponíveis
    if pred_proba is not None:
        fpr, tpr, _ = metrics.roc_curve(actual, pred_proba)
        auc = metrics.auc(fpr, tpr)
    else:
        fpr, tpr, _ = metrics.roc_curve(actual, pred)
        auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return (accuracy, f1, auc)

def mlflow_logging(model, X, y, name):
    with mlflow.start_run() as run:
        mlflow.set_tracking_uri("http://localhost:5000")
        run_id = run.info.run_id
        mlflow.set_tag('run_id', run_id)
        pred = model.predict(X)
        #metrics
        (accuracy, f1, auc) = eval_metrics(y, pred)
        #Logging the best parameters from grid search
        mlflow.log_params(model.best_params_)
        #Log the metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric('Mean CV score', model.best_score_)
        #Loggin artificial and model
        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model.best_estimator_, name)
        mlflow.end_run()

dataset, numerical_cols, categorical_cols = load_data(r"C:\Users\alvaro.careli\Documents\DeployMLModels\Mlops\Bootcamp_MlOps\MlOps2\MlFlow\train.csv")
X_train, X_test, y_train, y_test = preprocess_data(dataset, numerical_cols, categorical_cols)

rf = random_forest()
lr = logistic_regression()
dt = decision_tree()
mlflow_logging(rf, X_test, y_test, "random_forest_model")
mlflow_logging(lr, X_test, y_test, "logistic_regression_model")
mlflow_logging(dt, X_test, y_test, "decision_tree_model")