import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\dataset_filtrado_20_atributos.csv'
CAMINHO_SAIDA_FINAL = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\dataset_final_critico.csv'

# 1. Carregamento dos dados
df = pd.read_csv(CAMINHO_ENTRADA)

# ==========================================
# 2. TRATAMENTO CRÍTICO
# ==========================================
# Mantemos a remoção de nulos na variável mais crítica do seu domínio
df = df.dropna(subset=['bb04_idade_da_mae'])
print(f"Instâncias com idade nula removidas. Linhas restantes: {len(df)}")

df = df.dropna(subset=['k06_peso_engravidar'])
print(f"Instâncias com peso ao engravidar nula removidas. Linhas restantes: {len(df)}")

df = df.dropna(subset=['vd_imc_mae'])
print(f"Instâncias com imc nula removidas. Linhas restantes: {len(df)}")

df = df.dropna(subset=['k04_prenatal_semanas'])
print(f"Instâncias com inicio do prenatal nula removidas. Linhas restantes: {len(df)}")

df.to_csv(CAMINHO_SAIDA_FINAL, index=False, encoding='utf-8')
print(f"\nDataset com imputação MICE e One-Hot Encoding salvo em:\n{CAMINHO_SAIDA_FINAL}")

print("\nResumo do novo dataset:")
print(df.info())