import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\dataset_filtrado_20_atributos.csv'
CAMINHO_SAIDA_FINAL = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\dataset_final_ohe.csv'

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

# 2. Identificação das colunas
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()

# 3. Pré-processamento: Codificação de variáveis categóricas
encoder = OrdinalEncoder()
df_encoded = df.copy()
df_encoded[cat_cols] = encoder.fit_transform(df[cat_cols])

# 4. Aplicação da técnica MICE
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed_array = mice_imputer.fit_transform(df_encoded)

# 5. Reconstrução do DataFrame
df_imputed = pd.DataFrame(df_imputed_array, columns=df.columns)

# 6. Ajuste Fino: Tratamento pós-imputação
df_imputed[cat_cols] = df_imputed[cat_cols].round().astype(int)

# --- INÍCIO DA NOVA TRANSFORMAÇÃO ---

# Passo A: Reverter o OrdinalEncoder para recuperar as strings/categorias originais
categorias_recuperadas_array = encoder.inverse_transform(df_imputed[cat_cols])
df_cat_recup = pd.DataFrame(categorias_recuperadas_array, columns=cat_cols)

# ==========================================
# O PULO DO GATO: Isolando a Variável Alvo
# ==========================================
atributo_alvo = 'aleitamento_materno_exclusivo'

# Criamos uma lista apenas com as variáveis preditoras que precisam de OHE
cat_cols_features = [col for col in cat_cols if col != atributo_alvo]

# Passo B: Aplicar o OneHotEncoder APENAS nas variáveis preditoras
ohe = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')

# Ajusta e transforma apenas as colunas categóricas independentes
ohe_array = ohe.fit_transform(df_cat_recup[cat_cols_features])
ohe_feature_names = ohe.get_feature_names_out(cat_cols_features)

# Transforma em DataFrame
df_cat_ohe = pd.DataFrame(ohe_array, columns=ohe_feature_names)

# Passo C: Mesclar as colunas numéricas, as categóricas em OHE e o alvo intacto
df_numericas = df_imputed[num_cols].copy()
df_alvo = df_cat_recup[[atributo_alvo]].copy() # Pegamos o alvo recuperado (strings originais)

# Concatenamos lado a lado
df_final = pd.concat([df_numericas, df_cat_ohe, df_alvo], axis=1)

# 7. Salvamento do Dataset Final
df_final.to_csv(CAMINHO_SAIDA_FINAL, index=False, encoding='utf-8')
print(f"\nDataset com imputação MICE e One-Hot Encoding salvo em:\n{CAMINHO_SAIDA_FINAL}")

print("\nResumo do novo dataset:")
print(df_final.info())