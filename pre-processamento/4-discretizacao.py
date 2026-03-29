import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_sem_nulos.csv'
CAMINHO_SAIDA   = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_discretizado.csv'

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

# ==========================================
# 3. LIMPEZA DE REDUNDÂNCIAS
# ==========================================
# Remover variáveis numéricas que já possuem versão categórica pronta no dataset
colunas_para_remover = ['h02_peso', 'q06_renda', 'vd_ebia_escore', 'tempo_1a_amament']
df = df.drop(columns=colunas_para_remover)
print("✅ Variáveis numéricas redundantes descartadas.")

# ==========================================
# 4. DISCRETIZAÇÃO DAS VARIÁVEIS NUMÉRICAS
# ==========================================
# A. Idade da Mãe
df['idade_mae_cat'] = pd.cut(
    df['bb04_idade_da_mae'], 
    bins=[9, 19, 34, 100], 
    labels=['Adolescente', 'Adulta', 'Idade Avançada']
)

# B. Idade do Filho (em meses)
df['idade_filho_cat'] = pd.cut(
    df['idade_filho'], 
    bins=[0, 6, 12, 100], 
    labels=['0 a 6 meses', '6 a 12 meses', 'Mais de 1 ano'],
    include_lowest=True
)

# C. Número de Gestações e Filhos Vivos
limites_filhos = [1, 1.5, 3.5, 50]
rotulos_filhos = ['1', '2 a 3', '4 ou mais']

df['gestacoes_cat'] = pd.cut(df['k01_gestacoes'], bins=limites_filhos, labels=rotulos_filhos, include_lowest=True)
df['filhos_vivos_cat'] = pd.cut(df['k02_filhos_vivos'], bins=limites_filhos, labels=rotulos_filhos, include_lowest=True)

# D. Remover as numéricas originais que acabaram de ser discretizadas
# NOTA: Manteremos k28_rec como número até descobrirmos seu significado clínico
df = df.drop(columns=['bb04_idade_da_mae', 'idade_filho', 'k01_gestacoes', 'k02_filhos_vivos'])
print("✅ Discretização das idades e histórico materno concluída.")

# ==========================================
# 5. SALVAR DADOS TRANSFORMADOS
# ==========================================
df.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
print(f"\n🚀 Dataset 100% discretizado salvo em:\n{CAMINHO_SAIDA}")