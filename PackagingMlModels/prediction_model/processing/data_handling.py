"""
Carregar dados (de arquivos CSV, bancos de dados, APIs, etc.)
Pré-processar dados (limpeza, tratamento de valores ausentes, conversão de tipos)
Dividir dados em treino/teste/validação
Transformações específicas (normalização, padronização, encoding de variáveis categóricas)
Funções utilitárias para manipulação de datasets

"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
from PackagingMlModels.prediction_model.config import config

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
print(PACKAGE_ROOT)


#Carregando dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    _data.columns = [c.strip() for c in _data.columns] #removendo espacos em branco dos nomes das colunas
    return _data[config.FEATURES]

#Separando o dataset em treino e teste
def separate_data(data):
    X = data.drop(config.TARGET, axis = 1)
    y = data[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#Serialization
def save_pipeline(pipeline_to_save, file_name):
    save_path = os.path.join(config.SAVE_MODEL_PATH, file_name)
    print(save_path)
    joblib.dump(pipeline_to_save, save_path)
    print(f'Pipeline salvo em: {save_path}')

#Deserialization
def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH, pipeline_to_load)
    model_loaded = joblib.load(save_path)
    print('Model carregado com sucesso')
    return model_loaded
