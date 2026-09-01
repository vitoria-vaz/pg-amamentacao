import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Habilitando o IterativeImputer (necessário pois a feature ainda é experimental no sklearn)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS
# ==========================================
CAMINHO_ENTRADA = 'selecao_limpeza/dataset_amamentacao_filtrado.csv'
CAMINHO_SAIDA   = 'selecao_limpeza/mice/dataset_amamentacao_pronto_sem_discretizar.csv'

df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

# ==========================================
# 2. TRATAMENTO CRÍTICO
# ==========================================
# Mantemos a remoção de nulos na variável mais crítica do seu domínio
df = df.dropna(subset=['bb04_idade_da_mae'])
print(f"Instâncias com idade nula removidas. Linhas restantes: {len(df)}")

# ==========================================
# 3. IMPUTAÇÃO PREDITIVA COM MICE
# ==========================================
print("\n" + "="*50)
print("INICIANDO IMPUTAÇÃO MULTIVARIADA (MICE)")
print("="*50)

# Definimos todas as variáveis que farão parte do modelo (preditoras e alvos)
cols_mice = [
    'q07_renda_faixa', 'a00_regiao', 'bb04_idade_da_mae', 
    'k02_filhos_vivos', 'inic_prenat', 'num_consultas'
]

# O Scikit-Learn exige dados numéricos. Usaremos o OrdinalEncoder.
encoder = OrdinalEncoder()

# Criamos uma cópia para não alterar o df original durante a transformação
df_mice = df[cols_mice].copy()

# Mapeamos os dados conhecidos para numéricos (ignorando NaN temporariamente)
# O OrdinalEncoder padrão não lida bem com NaNs, então isolamos as colunas
for col in cols_mice:
    mascara_nao_nulos = df_mice[col].notnull()
    df_mice.loc[mascara_nao_nulos, col] = encoder.fit_transform(
        df_mice.loc[mascara_nao_nulos, [col]]
    ).ravel()

# Configurando o MICE
# Usamos o DecisionTreeClassifier para garantir saídas discretas (categorias)
imputer = IterativeImputer(
    estimator=DecisionTreeClassifier(random_state=42),
    initial_strategy='most_frequent', # Usa a moda na primeira iteração (substituição temporária)[cite: 10]
    max_iter=10,                      # O padrão é 10 iterações, iterando até a convergência[cite: 10]
    random_state=42
)

# Treinamos e imputamos todos os nulos de uma só vez!
print("Aplicando o algoritmo MICE (Decision Tree)...")
df_mice_imputado = imputer.fit_transform(df_mice)

# Substituímos os valores imputados de volta no dataframe original[cite: 10]
df_mice_imputado = pd.DataFrame(df_mice_imputado, columns=cols_mice)

# Revertemos o encoding (de volta para as strings/categorias originais)
for i, col in enumerate(cols_mice):
    df[col] = encoder.fit(df[[col]].dropna()).inverse_transform(
        df_mice_imputado[[col]]
    ).ravel()

print("✅ Todos os valores nulos preenchidos utilizando a relação entre todas as variáveis!")

# ==========================================
# 4. VERIFICAÇÃO FINAL E SALVAMENTO
# ==========================================
total_nulos_restantes = df.isnull().sum().sum()
print("\n" + "="*50)
if total_nulos_restantes == 0:
    df.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
    print(f"🚀 Dataset PRONTO PARA A REDE BAYESIANA (0 nulos) salvo em:\n{CAMINHO_SAIDA}")
else:
    print(f"⚠️ ATENÇÃO: Ainda existem {total_nulos_restantes} valores nulos no dataset.")
print("="*50)