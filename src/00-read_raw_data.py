import pandas as pd
import numpy as np
import os

#! Constantes de endereço necessárias para melhor organização de diretórios e pastas de trabalho.
#
SRC_DIR = os.path.join( os.path.abspath('..'), 'src')
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMGS_DIR = os.path.join(BASE_DIR, 'imgs')

input_file = 'recalque_superficial.xlsx'
input_path = os.path.join(DATA_DIR, input_file)

colunas = ['Caso', 's (mm)', 'L (m)', 'B (m)', 'Df (m)', 'q (kPa)', 'Nspt', 
'ANJOS']

data = pd.read_excel(input_path,
                      sheet_name = 'Planilha1',
                      header = 1,
                      usecols = colunas,
                      index_col = 0)

data.to_csv(os.path.join(DATA_DIR, "dados_recalque.csv"))
