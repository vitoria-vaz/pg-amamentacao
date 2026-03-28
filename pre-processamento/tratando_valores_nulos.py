import pandas as pd

# 1. Carregar o dataset
caminho_arquivo = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_filtrado.csv'
df = pd.read_csv(caminho_arquivo, encoding='utf-8')

# ==========================================
# 2. TRATAMENTO DOS NULOS
# ==========================================

# A. Excluir colunas com quase 100% de nulos
df = df.drop(columns=['k09_licenca', 'k10_meses'])
print("Colunas 'k09_licenca' e 'k10_meses' removidas com sucesso!")

# B. Tratar ausências pequenas nas numéricas (preenchendo com a Mediana)
df['bb04_idade_da_mae'] = df['bb04_idade_da_mae'].fillna(df['bb04_idade_da_mae'].median())
df['k28_rec'] = df['k28_rec'].fillna(df['k28_rec'].median())

# C. Tratar ausências moderadas nas categóricas (criando a categoria "Desconhecido")
df['inic_prenat'] = df['inic_prenat'].fillna("Desconhecido")
df['num_consultas'] = df['num_consultas'].fillna("Desconhecido")

# ==========================================
# 3. VERIFICAÇÃO E SALVAMENTO
# ==========================================

# Verifica se sobrou algum nulo
total_nulos_restantes = df.isnull().sum().sum()
print(f"\nQuantidade total de valores nulos no dataset agora: {total_nulos_restantes}")

if total_nulos_restantes == 0:
    # Salva o novo dataset tratado para usarmos na Etapa 4
    caminho_salvar = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_sem_nulos.csv'
    df.to_csv(caminho_salvar, index=False, encoding='utf-8')
    print(f"Dataset limpo e sem nulos salvo em: {caminho_salvar}")