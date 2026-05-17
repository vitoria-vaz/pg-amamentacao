import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS
# ==========================================
# Lê a base DISCRETIZADA (Antes do tratamento de nulos do Passo 4)
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_discretizado.csv'

# Cria uma pasta separada para os gráficos de diagnóstico
DIRETORIO_SAIDA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\graficos_pre_imputacao'

if not os.path.exists(DIRETORIO_SAIDA):
    os.makedirs(DIRETORIO_SAIDA)

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
print(f"Carregando dados pré-imputação de: {CAMINHO_ENTRADA}")
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

# Verificação de nulos para o console
nulos_totais = df.isnull().sum()
print("\n📋 Relatório de Valores Faltantes (NaN) por Variável:")
print(nulos_totais[nulos_totais > 0])
print("-" * 50)

# Estilo visual limpo e acadêmico
sns.set_theme(style="whitegrid")

# ==========================================
# 3. GERAÇÃO DOS GRÁFICOS (Loop por Colunas)
# ==========================================
print("Gerando gráficos de diagnóstico (com exibição de valores ausentes)...")

for coluna in df.columns:
    plt.figure(figsize=(10, 6))
    
    # ---- Se a variável for NUMÉRICA ----
    if pd.api.types.is_numeric_dtype(df[coluna]):
        ax = sns.histplot(data=df, x=coluna, bins=15, kde=True, color='#C44E52')
        plt.title(f'Distribuição (Pré-Imputação): {coluna}', fontsize=14, fontweight='bold')
        plt.ylabel('Frequência')
        plt.xlabel(coluna)
        
    # ---- Se a variável for CATEGÓRICA ----
    else:
        # O PULO DO GATO: Substitui os NaNs temporariamente por um texto visível 
        # para que eles apareçam como uma barra vermelha no gráfico!
        df_temp = df[coluna].copy()
        if df_temp.isnull().any():
            df_temp = df_temp.fillna('VALOR AUSENTE (NaN)')
            
        contagem = df_temp.value_counts()
        
        # Cria uma paleta de cores. Se a barra for 'VALOR AUSENTE (NaN)', pinta de vermelho para destacar.
        cores = ['#D62728' if idx == 'VALOR AUSENTE (NaN)' else '#4C72B0' for idx in contagem.index]
        
        # Gráfico de barras
        ax = sns.barplot(x=contagem.index, y=contagem.values, palette=cores)
        plt.title(f'Distribuição (Pré-Imputação): {coluna}', fontsize=14, fontweight='bold')
        plt.ylabel('Quantidade de Instâncias')
        plt.xlabel('Categorias')
        
        # Rotaciona os textos no eixo X
        plt.xticks(rotation=45, ha='right')
        
        # Adiciona o número exato no topo de cada barra
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 8), 
                        textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Ajusta as margens
    plt.tight_layout()
    
    # Salva o arquivo
    nome_arquivo = f"dist_pre_{coluna}.jpg"
    caminho_arquivo = os.path.join(DIRETORIO_SAIDA, nome_arquivo)
    plt.savefig(caminho_arquivo, dpi=300)
    plt.close() 
    
    print(f"📊 Gráfico gerado: {nome_arquivo}")

print(f"\n🚀 SUCESSO! Gráficos de diagnóstico salvos na pasta '{DIRETORIO_SAIDA}'.")