import pandas as pd
import matplotlib.pyplot as plt

caminho_arquivo = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_filtrado.csv'

df = pd.read_csv(caminho_arquivo, encoding='utf-8')

# 1. Resumo dos tipos de dados identificados pelo Pandas
print("Resumo dos tipos de dados identificados pelo Pandas:")
df.info() 

# ====================================================================================
# 2. Gráfico de barras para visualizar a quantidade de valores nulos por atributo

# 2.1. Configurar o tamanho da figura
plt.figure(figsize=(12, 6))

# 2.2. Criar o gráfico de barras
df.isnull().sum().sort_values(ascending=False).plot(kind='bar')

# 2.3. Adicionar títulos e rótulos
plt.title('Quantidade de Valores Nulos por Atributo')
plt.xlabel('Atributos')
plt.ylabel('Quantidade de Valores Nulos')
plt.xticks(rotation=90)
plt.tight_layout() 


# 2.4. Gerar jpg do gráfico (usando bbox_inches='tight' como garantia extra)
plt.savefig('contagem_valores_nulos.jpg', bbox_inches='tight')

# 2.5. Fechar a figura da memória (boa prática!)
plt.close()
# ====================================================================================

# 2. Isolar as colunas baseadas no tipo de dado (dtype)
# 'number' pega int64 e float64. 
cols_numericas = df.select_dtypes(include=['number']).columns.tolist()

# 'object' pega strings, 'category' para tipos categoricos do pandas e 'bool' para booleanos
cols_categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# 3. Exibindo os resultados
print(f"Total de Variáveis Numéricas: {len(cols_numericas)}")
print(cols_numericas)
print("-" * 50)
print(f"Total de Variáveis Categóricas: {len(cols_categoricas)}")
print(cols_categoricas)