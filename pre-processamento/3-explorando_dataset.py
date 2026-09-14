import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\dataset_filtrado_20_atributos.csv'
CAMINHO_ENTRADA2 = 'selecao_limpeza/dataset_amamentacao.csv'

# Nomes dos arquivos de imagem gerados
CAMINHO_GRAFICO_NULOS_TODOS = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\graficos_distribuicao\tentativa_2\contagem_valores_nulos_todos.jpg'
#CAMINHO_GRAFICO_NULOS_TODOS_2 = 'contagem_valores_nulos_todos_2.jpg'
#CAMINHO_GRAFICO_NULOS_FILTRADOS = 'contagem_valores_nulos_filtrados.jpg'
CAMINHO_GRAFICO_TIPOS = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\graficos_distribuicao\tentativa_2\contagem_tipos_dados.jpg'

# ==========================================
# 2. CARREGAMENTO DOS DADOS E INFORMAÇÕES
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')
df2 = pd.read_csv(CAMINHO_ENTRADA2, encoding='utf-8')

print("Resumo dos tipos de dados identificados pelo Pandas:")
df.info() 
print("\n" + "="*50 + "\n")

# ==========================================
# 3. ANÁLISE EXPLORATÓRIA: VALORES NULOS
# ==========================================

# ---------------------------------------------------------
# 3.1 Gráfico 1: Todos os atributos do dataset filtrado(Visão Geral)
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
df.isnull().sum().sort_values(ascending=False).plot(kind='bar', color='#4C72B0', edgecolor='black')

plt.title('Quantidade de Valores Nulos (Todos os Atributos)', fontsize=14, fontweight='bold')
plt.xlabel('Atributos', fontsize=12)
plt.ylabel('Quantidade de Valores Nulos', fontsize=12)
plt.xticks(rotation=90)

plt.tight_layout() 
plt.savefig(CAMINHO_GRAFICO_NULOS_TODOS, bbox_inches='tight')
plt.close()
print(f"Gráfico geral de nulos salvo como '{CAMINHO_GRAFICO_NULOS_TODOS}'!")

# ---------------------------------------------------------
# 3.2 Gráfico 2: Todos os atributos do dataset bruto (Visão Geral)
# ---------------------------------------------------------


# ---------------------------------------------------------
# 3.3 Gráfico 3: Apenas atributos com valores nulos (Zoom)
# ---------------------------------------------------------


# ==========================================
# 4. SEPARAÇÃO DOS TIPOS DE DADOS
# ==========================================
cols_numericas = df.select_dtypes(include=['number']).columns.tolist()
cols_categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print("\n" + "-" * 50)
print(f"Total de Variáveis Numéricas: {len(cols_numericas)}")
print(cols_numericas)
print("-" * 50)
print(f"Total de Variáveis Categóricas: {len(cols_categoricas)}")
print(cols_categoricas)
print("-" * 50)

# ==========================================
# 5. ANÁLISE EXPLORATÓRIA: GRÁFICO DOS TIPOS
# ==========================================
quantidades = [len(cols_numericas), len(cols_categoricas)]
nomes_tipos = ['Numéricas', 'Categóricas']

plt.figure(figsize=(8, 6))
barras = plt.bar(nomes_tipos, quantidades, color=['#55A868', '#4C72B0'], edgecolor='black', width=0.6)

plt.title('Distribuição de Variáveis por Tipo de Dado', fontsize=14, fontweight='bold')
plt.xlabel('Tipo de Variável', fontsize=12)
plt.ylabel('Quantidade', fontsize=12)

# Adicionar os números exatos em cima de cada barra
for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2., altura + 0.5,
             f'{int(altura)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout() 
plt.savefig(CAMINHO_GRAFICO_TIPOS, bbox_inches='tight')
plt.close()
print(f"Gráfico de contagem de tipos salvo como '{CAMINHO_GRAFICO_TIPOS}'!")