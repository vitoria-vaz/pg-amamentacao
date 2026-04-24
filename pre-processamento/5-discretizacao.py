import pandas as pd

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
colunas_para_remover = ['vd_ebia_escore', 'tempo_1a_amament']

# Só tenta remover se elas ainda existirem na base
colunas_existentes = [col for col in colunas_para_remover if col in df.columns]
df = df.drop(columns=colunas_existentes)
print("✅ Variáveis numéricas redundantes descartadas.")

# ==========================================
# 4. DISCRETIZAÇÃO DAS VARIÁVEIS NUMÉRICAS
# ==========================================
# A. Idade da Mãe
df['idade_mae_cat'] = pd.cut(
    df['bb04_idade_da_mae'], 
    bins=[9, 19, 34, 100], 
    labels=['Adolescente', 'Jovem Adulta', 'Adulta']
)

# C. Número de Gestações e Filhos Vivos
limites_filhos = [0.9, 1.5, 3.5, 50]
rotulos_filhos = ['Primípara (1)', 'Multípara (2 ou mais)', 'Grand Multípara (4 ou mais)'] #Talvez grand multipara?

df['gestacoes_cat'] = pd.cut(df['k01_gestacoes'], bins=limites_filhos, labels=rotulos_filhos, include_lowest=True)
df['filhos_vivos_cat'] = pd.cut(df['k02_filhos_vivos'], bins=limites_filhos, labels=rotulos_filhos, include_lowest=True)

# D. Remover as numéricas originais que acabaram de ser discretizadas
df = df.drop(columns=['bb04_idade_da_mae', 'idade_filho', 'k01_gestacoes', 'k02_filhos_vivos'])
print("✅ Discretização das idades e histórico materno concluída.")

# ==========================================
# 5. ATENÇÃO: REDE BAYESIANA NÃO ACEITA NÚMEROS!
# ==========================================
# Se k28_rec ainda estiver no dataset e for contínua, temos que excluí-la ou criar faixas.
if 'k28_rec' in df.columns:
    df = df.drop(columns=['k28_rec'])
    print("⚠️ Coluna k28_rec removida pois Redes Bayesianas exigem variáveis 100% categóricas.")

# ==========================================
# 6. SALVAR DADOS TRANSFORMADOS
# ==========================================
df.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
print(f"\n🚀 Dataset 100% discretizado e pronto para modelagem salvo em:\n{CAMINHO_SAIDA}")