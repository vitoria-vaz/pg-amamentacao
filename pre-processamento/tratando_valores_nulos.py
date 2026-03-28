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
# 3. PRÉ-PROCESSAMENTO (Tratamento de Nulos)
# ==========================================

# A. Excluir colunas com excesso de nulos (quase 100%)
colunas_para_remover = ['k09_licenca', 'k10_meses']
df = df.drop(columns=colunas_para_remover)
print(f"Colunas {colunas_para_remover} removidas com sucesso!")

# B. Tratar ausências pequenas nas numéricas
# NOTA: Preenchendo com a mediana para manter o máximo de dados possível, 
# mas futuramente pode-se avaliar a exclusão destas instâncias.
df['bb04_idade_da_mae'] = df['bb04_idade_da_mae'].fillna(df['bb04_idade_da_mae'].median())
df['k28_rec'] = df['k28_rec'].fillna(df['k28_rec'].median())

# C. Tratar ausências moderadas nas categóricas (criando categoria "Desconhecido")
df['inic_prenat'] = df['inic_prenat'].fillna("Desconhecido")
df['num_consultas'] = df['num_consultas'].fillna("Desconhecido")

# ==========================================
# 4. VERIFICAÇÃO E SALVAMENTO
# ==========================================
total_nulos_restantes = df.isnull().sum().sum()
print(f"\nQuantidade total de valores nulos no dataset agora: {total_nulos_restantes}")

if total_nulos_restantes == 0:
    df.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
    print(f"Dataset limpo e sem nulos salvo em:\n{CAMINHO_SAIDA}")
else:
    print("ATENÇÃO: Ainda existem valores nulos no dataset. O arquivo não foi salvo.")