"""
Carregar dados (de arquivos CSV, bancos de dados, APIs, etc.)
Pré-processar dados (limpeza, tratamento de valores ausentes, conversão de tipos)
Dividir dados em treino/teste/validação
Transformações específicas (normalização, padronização, encoding de variáveis categóricas)
Funções utilitárias para manipulação de datasets

"""
import os
import pandas as pd
from Mlops.Bootcamp_MlOps.PackagingMlModels.prediction_model.config import config
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
from prediction_model.config import config
print(PACKAGE_ROOT)

#Função para carregar o dataset 
def load_dataset(file_name, train = True):
    filepath = os.path.join(config.DATAPATH, file_name) #Vai unir os arquivos de config.DATAPATH com o nome do file_name
    _data = pd.read_csv(filepath)
    _data.columns = [c.strip() for c in _data.columns] #Retirando o espaços desnecessarios 
    # Retorna apenas as colunas do DataFrame que estão listadas em config.FEATURES,
    # garantindo que só as variáveis de interesse sejam usadas no restante do pipeline.
    return _data[config.FEATURES]

#Separando x e y
def separete_data(data):
    X = data.drop(config.TARGET, axis = 1)
    y = data[config.TARGET]
    return X, y 
