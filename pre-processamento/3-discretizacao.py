import pandas as pd

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_filtrado.csv'
CAMINHO_SAIDA   = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_discretizado.csv'

df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

# ==========================================
# 2. LIMPEZA DE REDUNDÂNCIAS
# ==========================================
# o que é o atributo vd_ebia_escore? deve remover ele? como discretizar ele?
colunas_para_remover = ['vd_ebia_escore', 'tempo_1a_amament']
colunas_existentes = [col for col in colunas_para_remover if col in df.columns]
df = df.drop(columns=colunas_existentes)
print("✅ Variáveis numéricas redundantes descartadas.")

# ==========================================
# 3. DISCRETIZAÇÃO DAS VARIÁVEIS NUMÉRICAS
# ==========================================
# A. Idade da Mãe (Conforme orientação das pesquisadoras)
df['idade_mae_cat'] = pd.cut(
    df['bb04_idade_da_mae'], 
    bins=[9, 19, 29, 34, 110], 
    labels=['Adolescente', 'Jovem Adulta', 'Adulta', 'Maternidade Tardia']
)

# B. Número de Gestações e Filhos Vivos (Mantido binário)
limites_filhos = [0.9, 1.5, 50]
rotulos_filhos = ['Primípara (1)', 'Multípara (2 ou mais)'] 

df['gestacoes_cat'] = pd.cut(df['k01_gestacoes'], bins=limites_filhos, labels=rotulos_filhos, include_lowest=True)
df['filhos_vivos_cat'] = pd.cut(df['k02_filhos_vivos'], bins=limites_filhos, labels=rotulos_filhos, include_lowest=True)

# C. Remover as numéricas originais (e a Idade do Filho e K28, que não vamos usar)
df = df.drop(columns=['bb04_idade_da_mae', 'k01_gestacoes', 'k02_filhos_vivos'])

print("✅ Discretização das idades e histórico materno concluída.")

df.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
print(f"🚀 Dataset discretizado salvo em:\n{CAMINHO_SAIDA}")