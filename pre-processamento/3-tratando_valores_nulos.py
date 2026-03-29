import pandas as pd

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_filtrado.csv'
CAMINHO_SAIDA   = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_sem_nulos.csv'

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

# ==========================================
# 3. TRATAMENTO DE OUTLIERS E ANOMALIAS
# ==========================================
# Substituir o código 99 (Não sabe/Ignorado) pela mediana 
# NOTA: A mediana é calculada ignorando os próprios 99s para não ser enviesada.
mediana_gest = df.loc[df['k01_gestacoes'] != 99, 'k01_gestacoes'].median()
mediana_filhos = df.loc[df['k02_filhos_vivos'] != 99, 'k02_filhos_vivos'].median()

df['k01_gestacoes'] = df['k01_gestacoes'].replace(99, mediana_gest)
df['k02_filhos_vivos'] = df['k02_filhos_vivos'].replace(99, mediana_filhos)
print("✅ Outliers e códigos de erro (99) tratados com sucesso!")

# ==========================================
# 4. TRATAMENTO DE VALORES NULOS
# ==========================================
# A. Excluir colunas com excesso de nulos (quase 100%)
colunas_para_remover = ['k09_licenca', 'k10_meses']
df = df.drop(columns=colunas_para_remover)
print(f"✅ Colunas vazias {colunas_para_remover} removidas.")

# B. Tratar ausências pequenas nas numéricas (agora com a base limpa de outliers)
df['bb04_idade_da_mae'] = df['bb04_idade_da_mae'].fillna(df['bb04_idade_da_mae'].median())
df['k28_rec'] = df['k28_rec'].fillna(df['k28_rec'].median())

# C. Tratar ausências moderadas nas categóricas
df['inic_prenat'] = df['inic_prenat'].fillna("Desconhecido")
df['num_consultas'] = df['num_consultas'].fillna("Desconhecido")
print("✅ Valores nulos preenchidos com sucesso!")

# ==========================================
# 5. VERIFICAÇÃO E SALVAMENTO
# ==========================================
total_nulos_restantes = df.isnull().sum().sum()

if total_nulos_restantes == 0:
    df.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
    print(f"\n🚀 Dataset limpo, sem outliers e sem nulos salvo em:\n{CAMINHO_SAIDA}")
else:
    print(f"\n⚠️ ATENÇÃO: Ainda existem {total_nulos_restantes} valores nulos no dataset. Arquivo não salvo.")