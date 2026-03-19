import logging 
import os
from loggin.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

#Constantes para config do log
LOG_DIR = 'logs'
LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d")}.log'
MAX_LOG_SIZE = 5 * 1024 * 1024 # 5 MB
BACKUP_COUNT = 3 # Number of backup log files to keep 


#Construct log file path
log_dir_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok = True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    #Definindo o formater 
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    #Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    #Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
configure_logger()