import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_sem_nulos.csv'
CAMINHO_SAIDA   = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_discretizado.csv'
CAMINHO_GRAFICO = 'distribuicao_idade_mae.jpg'

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

# ==========================================
# 3. PRÉ-PROCESSAMENTO (Limpeza e Discretização)
# ==========================================
# Remover variáveis numéricas redundantes
df = df.drop(columns=['h02_peso', 'q06_renda'])

# Discretizar a Idade da Mãe
limites_idade = [9, 19, 34, 100]
rotulos_idade = ['Adolescente', 'Adulta', 'Idade Avançada']

df['idade_mae_cat'] = pd.cut(df['bb04_idade_da_mae'], bins=limites_idade, labels=rotulos_idade)

# Descartar a numérica original após criar a categórica
df = df.drop(columns=['bb04_idade_da_mae'])
print("Variáveis originais removidas e idade da mãe discretizada com sucesso!")

# ==========================================
# 4. SALVAR DADOS TRANSFORMADOS
# ==========================================
df.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
print(f"Dataset discretizado salvo em: {CAMINHO_SAIDA}\n")

# ==========================================
# 5. ANÁLISE EXPLORATÓRIA (Gráficos)
# ==========================================
# Contar e ordenar as categorias
contagem_idade = df['idade_mae_cat'].value_counts().sort_index()

# Plotar gráfico
plt.figure(figsize=(8, 6))
contagem_idade.plot(kind='bar', color=['#55A868', '#4C72B0', '#C44E52'], edgecolor='black')

plt.title('Distribuição da Idade das Mães', fontsize=14, fontweight='bold')
plt.xlabel('Faixa Etária', fontsize=12)
plt.ylabel('Quantidade de Mães', fontsize=12)
plt.xticks(rotation=0) 

plt.tight_layout()
plt.savefig(CAMINHO_GRAFICO, bbox_inches='tight')
plt.close()

print(f"Gráfico gerado e salvo com sucesso como '{CAMINHO_GRAFICO}'!")