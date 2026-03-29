import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_discretizado.csv'
CAMINHO_GRAFICO_CALOR = 'mapa_calor_cramers_v.jpg'

# ==========================================
# 2. FUNÇÃO MATEMÁTICA: V DE CRAMÉR
# ==========================================
def cramers_v(x, y):
    """ Calcula a estatística V de Cramér para duas variáveis categóricas. """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    # Correção de viés (Bias correction) para melhorar a precisão
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    # Prevenção de divisão por zero
    denominator = min((kcorr-1), (rcorr-1))
    if denominator == 0:
        return 0.0
    
    return np.sqrt(phi2corr / denominator)

# ==========================================
# 3. CARREGAMENTO DOS DADOS
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

# Selecionar apenas as colunas categóricas (o que deve ser 100% delas a essa altura)
df_cat = df.select_dtypes(include=['object', 'category', 'bool'])
colunas = df_cat.columns.tolist()

print(f"Calculando o V de Cramér para {len(colunas)} variáveis. Isso pode levar alguns segundos...")

# ==========================================
# 4. CONSTRUÇÃO DA MATRIZ DE CORRELAÇÃO
# ==========================================
# Criar um DataFrame vazio para armazenar os resultados
matriz_cramer = pd.DataFrame(np.zeros((len(colunas), len(colunas))), index=colunas, columns=colunas)

# Preencher a matriz com os cálculos par a par
for col1 in colunas:
    for col2 in colunas:
        matriz_cramer.loc[col1, col2] = cramers_v(df_cat[col1], df_cat[col2])

print("Matriz calculada com sucesso! Gerando o Heatmap...")

# ==========================================
# 5. VISUALIZAÇÃO: MAPA DE CALOR (HEATMAP)
# ==========================================
# Configurar um tamanho de figura bem grande, pois temos muitas variáveis
plt.figure(figsize=(24, 20))

# Criar o heatmap usando a biblioteca Seaborn
# cmap='Blues' deixa tons mais escuros para correlações mais fortes
sns.heatmap(matriz_cramer, annot=False, cmap='Blues', fmt=".2f", 
            linewidths=0.5, cbar_kws={"shrink": .8})

plt.title("Mapa de Calor: Associação V de Cramér entre Atributos", fontsize=20, fontweight='bold', pad=20)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Ajustar layout e salvar em alta resolução (dpi=300)
plt.tight_layout()
plt.savefig(CAMINHO_GRAFICO_CALOR, dpi=300, bbox_inches='tight')
plt.close()

print(f"Mapa de calor gerado e salvo em altíssima resolução como '{CAMINHO_GRAFICO_CALOR}'!")