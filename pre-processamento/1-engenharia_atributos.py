import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao.csv'
CAMINHO_SAIDA   = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_eng.csv'
DISTRIBUICAO_ALVO = 'distribuicao_alvo.jpg'

# Carrega o dado bruto
df = pd.read_csv(CAMINHO_ENTRADA)

# ==========================================
# TRATAMENTO INICIAL DE NULOS CRÍTICOS
# ==========================================
# Removendo as 7 instâncias nulas da idade da mãe
df = df.dropna(subset=['bb04_idade_da_mae'])
print(f"✅ 7 instâncias com idade da mãe nula foram removidas. Linhas restantes: {len(df)}")

# ==========================================
# NOVAS CATEGORIZAÇÕES DE ATRIBUTOS
# ==========================================

# A. Agrupamento de Escolaridade (j10_serie) para diminuir dimensionalidade
mapeamento_escolaridade = {
    'Sem estudo': 'Fundamental I',
    '1° ano do ensino fundamental': 'Fundamental I',
    '1ª série/ 2°ano do ensino fundamental': 'Fundamental I',
    '2ª série/ 3°ano do ensino fundamental': 'Fundamental I',
    '3ª série/ 4°ano do ensino fundamental': 'Fundamental I',
    '4ª série/ 5°ano do ensino fundamental': 'Fundamental I',
    '5ª série/ 6°ano do ensino fundamental': 'Fundamental II',
    '6ª série/ 7°ano do ensino fundamental': 'Fundamental II',
    '7ª série/ 8°ano do ensino fundamental': 'Fundamental II',
    '8ª série/ 9°ano do ensino fundamental': 'Fundamental II',
    '1°ano do ensino médio': 'Ensino Médio',
    '2°ano do ensino médio': 'Ensino Médio',
    '3°ano do ensino médio': 'Ensino Médio',
    'Ensino superior incompleto': 'Superior',
    'Ensino superior completo': 'Superior'
}
df['escolaridade_cat'] = df['j10_serie'].map(mapeamento_escolaridade)

# B. Ajuste na faixa de renda (q07_renda_faixa) unindo as classes mais altas para estabilidade da Rede
df['q07_renda_faixa'] = df['q07_renda_faixa'].replace({
    'De R$ 5.001,00 até R$ 10.000,00': 'R$ 5.001,00 ou mais',
    'R$ 10.001,00 ou mais': 'R$ 5.001,00 ou mais'
})

# ==========================================
# PASSOS 2 e 3: Preparar atributo alvo
# ==========================================
def classificar_amamentacao_exclusiva(row):
    idade = row['idade_filho']
    tempo = row['k18_somente']
    medida = row['k19_somente_medida'] 
    
    # 1. TRATAMENTO DE VALORES INCONSISTENTES / AUSENTES
    if pd.isna(tempo) or pd.isna(medida):
        return 'Indeterminado'
        
    # 2. TRATAMENTO DE QUEM AINDA AMAMENTA (CÓDIGO 88)
    if tempo == 88.0:
        if idade >= 6:
            return 'Sucesso (AME 6m+)'
        else:
            return 'Censurado (Ainda amamenta < 6m)'
            
    # 3. PADRONIZAÇÃO DA UNIDADE DE TEMPO (Tudo para meses)
    tempo_em_meses = tempo
    if medida == 1.0 or medida == 'Dias':
        tempo_em_meses = tempo / 30.0  # Converte dias para meses de forma aproximada
        
    # 4. CLASSIFICAÇÃO FINAL DO ALVO
    if tempo_em_meses >= 6.0:
        return 'Sucesso (AME 6m+)'
    else:
        return 'Desmame precoce (< 6m)'
    
df['alvo_amamentacao'] = df.apply(classificar_amamentacao_exclusiva, axis=1)

# Verificação e Contagem do Alvo
counts = df['alvo_amamentacao'].value_counts()
print("\nDistribuição do Alvo criado:")
print(counts)

qtde_censurado = (df['alvo_amamentacao'] == 'Censurado (Ainda amamenta < 6m)').sum()
print(f"Quantidade de casos censurados (Ainda amamenta < 6m): {qtde_censurado}")

# Gerar Gráfico
plt.figure(figsize=(10, 6))
ax = counts.plot(kind='bar', color=['#4C72B0', '#C44E52', '#55A868', '#8172B2'], edgecolor='black')
plt.title('Distribuição da Variável Alvo (Antes da Filtragem)', fontsize=14, fontweight='bold')
plt.xlabel('Categoria', fontsize=12)
plt.ylabel('Quantidade de Instâncias', fontsize=12)
plt.xticks(rotation=45, ha='right')

for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 8), 
                textcoords='offset points', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(DISTRIBUICAO_ALVO, bbox_inches='tight')
plt.close()

# ==========================================
# PASSO 6a: Raça e Cor da mãe (j03_cor)
# ==========================================
# Ex: 'Parda (mulata, cabocla...)' vira apenas 'Parda'
df['raca_cor_mae'] = df['j03_cor'].str.split(' \(').str[0]

# ---------------------------------------------------------
# PASSO 8a: Remover mães que adotaram (d06 == 1)
# ---------------------------------------------------------
df = df[df['d06_relacao_responsavel'] != 1.0]

# ==========================================
# PASSO 13a: Início Precoce da Amamentação
# ==========================================
df['inicio_precoce_amamentacao'] = np.where(
    (df['k13_tempo_medida'] == 'Horas') & (df['k12_tempo'] <= 1),
    'Sim',
    'Não'
)

# ==========================================
# PASSO 14a: Introdução Precoce de Líquidos (K16)
# ==========================================
df['k16_liquido'] = df['k16_liquido'].replace({
    'Não sabe/Não quis responder': 'Desconhecido'
})

# ==========================================
# REMOÇÃO DE REDUNDÂNCIAS E ATRIBUTOS ORIGINAIS
# ==========================================
# Removemos as colunas originais que deram origem às novas features.
colunas_para_dropar = [
    'k09_licenca', 'k10_meses', 'k12_tempo', 'k13_tempo_medida', 
    'k18_somente', 'k19_somente_medida', 'd06_relacao_responsavel',
    'j10_serie', 'j03_cor', 'q06_renda', 'h02_peso'
]

df_transformado = df.drop(columns=[col for col in colunas_para_dropar if col in df.columns])

# Filtro final do alvo para treinamento
df_final = df_transformado[df_transformado['alvo_amamentacao'].isin(['Sucesso (AME 6m+)', 'Desmame precoce (< 6m)'])]

# Salvar
df_final.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
print(f"🚀 Pipeline de Engenharia atualizado com sucesso! Arquivo salvo em: {CAMINHO_SAIDA}")