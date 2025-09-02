import pathlib
import os 
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent.parent
# Configurações do projeto
#Entrando com o caminho do dataset
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets") #Entrando com o caminho e as pasta do dataset 
#Entrando com o nome do arquivo
FILE_NAME = 'loan_approval_dataset.csv'
TEST_FILE = 'test_data.csv'
#Nome do modelo a ser criado
MODEL_NAME = 'classification.pkl'
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
CAT_FEATURES = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']