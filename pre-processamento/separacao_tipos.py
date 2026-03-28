import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_filtrado.csv'
CAMINHO_GRAFICO_NULOS = 'contagem_valores_nulos.jpg'

# ==========================================
# 2. CARREGAMENTO DOS DADOS E INFORMAÇÕES
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

print("Resumo dos tipos de dados identificados pelo Pandas:")
df.info() 
print("\n" + "="*50 + "\n")

# ==========================================
# 3. ANÁLISE EXPLORATÓRIA: VALORES NULOS
# ==========================================
# Configurar o tamanho da figura
plt.figure(figsize=(12, 6))

# Criar o gráfico de barras ordenado
df.isnull().sum().sort_values(ascending=False).plot(kind='bar', color='#4C72B0', edgecolor='black')

# Adicionar títulos e rótulos
plt.title('Quantidade de Valores Nulos por Atributo', fontsize=14, fontweight='bold')
plt.xlabel('Atributos', fontsize=12)
plt.ylabel('Quantidade de Valores Nulos', fontsize=12)
plt.xticks(rotation=90)

# Ajustar layout para não cortar as margens
plt.tight_layout() 

# Gerar e salvar o gráfico
plt.savefig(CAMINHO_GRAFICO_NULOS, bbox_inches='tight')
plt.close() # Fechar a figura da memória
print(f"Gráfico de valores nulos salvo com sucesso como '{CAMINHO_GRAFICO_NULOS}'!")

# ==========================================
# 4. SEPARAÇÃO DOS TIPOS DE DADOS
# ==========================================
# 'number' isola variáveis int64 e float64
cols_numericas = df.select_dtypes(include=['number']).columns.tolist()

# 'object', 'category' e 'bool' isolam as variáveis categóricas
cols_categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Exibir os resultados da separação
print("\n" + "-" * 50)
print(f"Total de Variáveis Numéricas: {len(cols_numericas)}")
print(cols_numericas)
print("-" * 50)
print(f"Total de Variáveis Categóricas: {len(cols_categoricas)}")
print(cols_categoricas)
print("-" * 50)