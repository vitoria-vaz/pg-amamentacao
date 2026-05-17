import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS
# ==========================================
# Lê a base FINAL, que passou por filtragem, discretização e imputação de nulos
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_pronto.csv'

# Cria uma pasta separada para os gráficos finais validados
DIRETORIO_SAIDA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\graficos_pos_imputacao'

if not os.path.exists(DIRETORIO_SAIDA):
    os.makedirs(DIRETORIO_SAIDA)

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
print(f"Carregando dados validados de: {CAMINHO_ENTRADA}")
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

# Verificação de segurança: Confirmar se há nulos antes de plotar
nulos_totais = df.isnull().sum().sum()
print(f"Total de atributos: {df.shape[1]}")
print(f"Total de valores nulos na base: {nulos_totais}")

if nulos_totais > 0:
    print("⚠️ AVISO: Ainda existem nulos na base! Os gráficos serão gerados, mas revise o passo 4.")
else:
    print("✅ Base 100% limpa! Gerando os gráficos definitivos...\n")

# Estilo visual limpo e acadêmico
sns.set_theme(style="whitegrid")

# ==========================================
# 3. GERAÇÃO DOS GRÁFICOS (Loop por Colunas)
# ==========================================
for coluna in df.columns:
    plt.figure(figsize=(10, 6))
    
    # ---- Se a variável for NUMÉRICA ----
    if pd.api.types.is_numeric_dtype(df[coluna]):
        # Usa um histograma com curva de densidade
        ax = sns.histplot(data=df, x=coluna, bins=15, kde=True, color='#2C7BB6')
        plt.title(f'Distribuição (Final): {coluna}', fontsize=14, fontweight='bold')
        plt.ylabel('Frequência')
        plt.xlabel(coluna)
        
    # ---- Se a variável for CATEGÓRICA ----
    else:
        # Pega a contagem exata das categorias
        contagem = df[coluna].value_counts()
        
        # Gráfico de barras ordenado do maior para o menor
        ax = sns.barplot(x=contagem.index, y=contagem.values, palette='mako')
        plt.title(f'Distribuição (Final): {coluna}', fontsize=14, fontweight='bold')
        plt.ylabel('Quantidade de Mães / Bebês')
        plt.xlabel('Categorias')
        
        # Rotaciona os textos no eixo X se forem muito grandes (ex: Escolaridade)
        plt.xticks(rotation=45, ha='right')
        
        # Adiciona o número exato no topo de cada barra para facilitar a leitura no TCC
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 8), 
                        textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Ajusta as margens para não cortar as legendas
    plt.tight_layout()
    
    # Salva o arquivo como imagem JPG em alta resolução
    nome_arquivo = f"dist_final_{coluna}.jpg"
    caminho_arquivo = os.path.join(DIRETORIO_SAIDA, nome_arquivo)
    plt.savefig(caminho_arquivo, dpi=300)
    plt.close() # Libera a memória
    
    print(f"📊 Gráfico gerado: {nome_arquivo}")

print(f"\n🚀 SUCESSO! Todos os {df.shape[1]} gráficos finais foram salvos na pasta '{DIRETORIO_SAIDA}'.")