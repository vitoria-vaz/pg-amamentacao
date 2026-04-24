import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_filtrado.csv' # Atualize o caminho se necessário
DIRETORIO_SAIDA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\graficos_distribuicao'

# Cria o diretório para salvar os gráficos se ele não existir
if not os.path.exists(DIRETORIO_SAIDA):
    os.makedirs(DIRETORIO_SAIDA)

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
print(f"Carregando dados de: {CAMINHO_ENTRADA}")
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

print(f"Dataset carregado! Total de colunas: {df.shape[1]}")
print("Gerando gráficos. Isso pode levar alguns segundos...\n")

# Estilo bonito para os gráficos usando o Seaborn
sns.set_theme(style="whitegrid")

# ==========================================
# 3. GERAÇÃO DOS GRÁFICOS (Loop por Colunas)
# ==========================================
for coluna in df.columns:
    plt.figure(figsize=(10, 6))
    
    # ---- Se a variável for NUMÉRICA ----
    if pd.api.types.is_numeric_dtype(df[coluna]):
        # Usamos um histograma (displot) para variáveis contínuas
        sns.histplot(data=df, x=coluna, bins=20, kde=True, color='#4C72B0')
        plt.title(f'Distribuição Numérica: {coluna}', fontsize=14, fontweight='bold')
        plt.ylabel('Frequência')
        
    # ---- Se a variável for CATEGÓRICA (Texto/Booleano) ----
    else:
        # Pega a contagem de cada categoria, ignorando nulos no cálculo do ranking
        contagem = df[coluna].value_counts()
        
        # Cria um gráfico de barras
        ax = sns.barplot(x=contagem.index, y=contagem.values, palette='viridis')
        plt.title(f'Distribuição Categórica: {coluna}', fontsize=14, fontweight='bold')
        plt.ylabel('Quantidade')
        
        # Se os nomes das categorias forem longos, rotaciona os textos no eixo X
        plt.xticks(rotation=45, ha='right')
        
        # Adiciona o número exato em cima de cada barra
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Ajusta o layout para não cortar textos
    plt.tight_layout()
    
    # Salva o gráfico
    nome_arquivo = f"distribuicao_{coluna}.jpg"
    caminho_arquivo = os.path.join(DIRETORIO_SAIDA, nome_arquivo)
    plt.savefig(caminho_arquivo, dpi=200) # Alta resolução
    plt.close() # Fecha a figura para não consumir memória
    
    print(f"Gráfico salvo: {nome_arquivo}")

print(f"\nSucesso! Todos os {df.shape[1]} gráficos foram salvos na pasta '{DIRETORIO_SAIDA}'.")