import pathlib
import os 

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Configurações do projeto
#Entrando com o caminho do dataset
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets") #Entrando com o caminho e as pasta do dataset 
#Entrando com o nome do arquivo
FILE_NAME = 'loan_approval_dataset.csv'
TEST_FILE = 'test_data.csv'
#Nome do modelo a ser criado
MODEL_NAME = 'classification.pkl'
PIPELINE_FILE = 'classification.pkl'
#Indicando diretorio de salvamento do modelo
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')
#Indicando o target (variavel alvo)
TARGET = 'loan_status'
#Features finais que serao usadas no modelo
FEATURES = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
       'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value','loan_status']
#Pred features : features mas sem a variavel target 
PRED_FEATURES = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
       'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
#Variaveis numericas
NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
#VARIAVEIS CATEGORICAS
CAT_FEATURES = ['education', 'self_employed']
# Dict para transformacao
FEATURES_TO_ENCODE = {
    'education': ['Graduate'],
    'self_employed': ['Yes']
}

NEW_FEATURE_ADD = 'total_assets_value'
FEATURE_TO_ADD = ['residential_assets_value','commercial_assets_value', 'luxury_assets_value','bank_asset_value' ]

DROP_FEATURES = ['residential_assets_value','commercial_assets_value', 'luxury_assets_value','bank_asset_value' ]

LOG_FEATURES = ['income_annum','loan_amount','total_assets_value'] # taking log of numerical columns