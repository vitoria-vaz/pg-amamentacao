import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_filtrado.csv'

# Nomes dos arquivos de imagem gerados
CAMINHO_GRAFICO_NULOS_TODOS = 'contagem_valores_nulos_todos.jpg'
CAMINHO_GRAFICO_NULOS_FILTRADOS = 'contagem_valores_nulos_filtrados.jpg'
CAMINHO_GRAFICO_TIPOS = 'contagem_tipos_dados.jpg'

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

# ---------------------------------------------------------
# 3.1 Gráfico 1: Todos os atributos (Visão Geral)
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
# 3.2 Gráfico 2: Apenas atributos com valores nulos (Zoom)
# ---------------------------------------------------------
nulos_totais = df.isnull().sum()
nulos_filtrados = nulos_totais[nulos_totais > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = nulos_filtrados.plot(kind='bar', color='#C44E52', edgecolor='black')

plt.title('Atributos que possuem Valores Nulos', fontsize=14, fontweight='bold')
plt.xlabel('Atributos', fontsize=12)
plt.ylabel('Quantidade de Valores Nulos', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Adicionar os números exatos em cima de cada barra
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 8), 
                textcoords='offset points', fontsize=11, fontweight='bold')

plt.tight_layout() 
plt.savefig(CAMINHO_GRAFICO_NULOS_FILTRADOS, bbox_inches='tight')
plt.close()
print(f"Gráfico filtrado de nulos salvo como '{CAMINHO_GRAFICO_NULOS_FILTRADOS}'!")

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