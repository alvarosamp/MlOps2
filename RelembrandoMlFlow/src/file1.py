import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://localhost:5000")

# Carregar o dataset
wine = load_wine()
X = wine.data
y = wine.target

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Definindo os parametros do modelo
max_depth = 5
n_estimators = 8

#Definindo o nome do experimento
mlflow.set_experiment("Wine_Classification_1")

with mlflow.start_run():
    print('Iniciando o experimento...')
    rf= RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    print('treinando o modelo...')
    rf.fit(X_train, y_train)
    print('Modelo treinado, realizando as previsões...')
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Previsões realizadas, acurácia calculada: {acc}')
    #Logando os parametros e a métrica de acurácia
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    
    #Carregando a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    #Plotando a matriz de confusão
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    #Salvando o artefato gerado
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png", artifact_path="confusion_matrices")
    #Testando minha tag
    mlflow.set_tags({"autor": 'Alvaro', "proejeto": "Wine Classification"})
    #Logando o modelo
    mlflow.sklearn.log_model(rf, "Rf Model")
    print(f"Acurácia: {acc}")


