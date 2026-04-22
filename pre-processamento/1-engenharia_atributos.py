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
# PASSOS 2 e 3: Preparar atributo alvo
# ==========================================
def classificar_amamentacao_exclusiva(row):
    idade = row['idade_filho']
    tempo = row['k18_somente']
    medida = row['k19_somente_medida'] 
    
    # 1. TRATAMENTO DE VALORES INCONSISTENTES / AUSENTES
    # Se for nulo
    if pd.isna(tempo) or pd.isna(medida):
        return 'Indeterminado'
        
    # 2. TRATAMENTO DE QUEM AINDA AMAMENTA (CÓDIGO 88)
    if tempo == 88.0:
        if idade >= 6:
            return 'Sucesso (AME 6m+)'
        else:
            # ATENÇÃO AO PASSO 3A DA PROFESSORA:
            # Esse é o dado censurado! O bebê tem menos de 6m e ainda mama.
            # Não sabemos se ele vai chegar aos 6 meses.
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

# Count values
counts = df['alvo_amamentacao'].value_counts()
print(counts)

counts_2 = df['amament_cat'].value_counts()
print("Alvo amament_cat: ")
print(counts_2)

# Generate Plot
plt.figure(figsize=(10, 6))
ax = counts.plot(kind='bar', color=['#4C72B0', '#C44E52', '#55A868', '#8172B2'], edgecolor='black')
plt.title('Distribuição da Variável Alvo (Antes da Filtragem)', fontsize=14, fontweight='bold')
plt.xlabel('Categoria', fontsize=12)
plt.ylabel('Quantidade de Instâncias', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Add values on top of bars
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 8), 
                textcoords='offset points', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(DISTRIBUICAO_ALVO, bbox_inches='tight')
plt.close()

# ==========================================
# PASSO 4a: Atributo Renda (Apenas Categórico)
# ==========================================
# A base já possui a variável q07_renda_faixa que é perfeitamente categorizada.
# Portanto, vamos apenas remover a renda contínua (q06_renda).

df = df.drop(columns=['q06_renda'])

print("Variável de renda contínua removida. Mantida apenas a versão em faixas.")

# ==========================================
# PASSO 5a: Sumarização bicos artificiais (K24)
# ==========================================
# Você pode somar quantas opções a mãe marcou, ou criar um "Usou algum? Sim/Não"
colunas_bicos = [col for col in df.columns if 'k24' in col]
df['usou_bico_artificial'] = df[colunas_bicos].apply(lambda row: 'Sim' if 'Sim' in row.values else 'Não', axis=1)

# ==========================================
# PASSO 6a: Raça e Cor da mãe (j03_cor)
# ==========================================
# Vamos remover os textos explicativos entre parênteses para limpar o modelo
# Ex: 'Parda (mulata, cabocla...)' vira apenas 'Parda'

df['raca_cor_mae'] = df['j03_cor'].str.split(' \(').str[0]

# Removendo a coluna antiga suja
df = df.drop(columns=['j03_cor'])

print("Distribuição limpa de Raça/Cor:")
print(df['raca_cor_mae'].value_counts())

# ---------------------------------------------------------
# PASSO 8a: Remover mães que adotaram (d06 == 1)
# ---------------------------------------------------------
# d06_relacao_responsavel = 1 significa "Mãe adotiva"
df = df[df['d06_relacao_responsavel'] != 1.0]

# ==========================================
# PASSO 10a: Licença Maternidade >= 4 meses
# ==========================================
# Apenas 0,6% (55 mães) de toda a base de dados possuía preenchimento válido nessas questões, caracterizando uma variável de variância quase zero. 
# Para evitar o acréscimo de ruído e dimensionalidade desnecessária na Rede Bayesiana, optei por remover os atributos K09 e K10.

# ==========================================
# PASSO 13a: Início Precoce da Amamentação
# ==========================================
# Assumindo que k13 diz a unidade de medida e k12 diz o tempo
# Queremos "Sim" para quem mamou na primeira hora de vida
df['inicio_precoce_amamentacao'] = np.where(
    (df['k13_tempo_medida'] == 'Horas') & (df['k12_tempo'] <= 1),
    'Sim',
    'Não'
)

# ==========================================
# PASSO 14a: Introdução Precoce de Líquidos (K16)
# ==========================================
# Vamos simplificar a categoria de incerteza e manter Sim/Não intactos
df['k16_liquido'] = df['k16_liquido'].replace({
    'Não sabe/Não quis responder': 'Desconhecido'
})

print("Distribuição da variável K16 (Líquidos precocemente):")
print(df['k16_liquido'].value_counts())

# ==========================================
# PASSO 15a: Peso (Manter peso_cat e remover numérico)
# ==========================================
# Como o atributo categórico (peso_cat) já existe na base, 
# deletamos a versão numérica para não duplicar informação na Rede Bayesiana.

df = df.drop(columns=['h02_peso'])

print("Variável peso contínua descartada com sucesso. Usando peso_cat.")

# ==========================================
# DEPOIS de criar as novas, excluímos as antigas!
# ==========================================
# Adicionei k18, k19 e d06 na lista de remoção
colunas_para_dropar = [
    'k09_licenca', 'k10_meses', 'k12_tempo', 'k13_tempo_medida', 
    'k18_somente', 'k19_somente_medida', 'd06_relacao_responsavel'
] + colunas_bicos

df_transformado = df.drop(columns=colunas_para_dropar)

# ==========================================
# PASSO 1a: SELEÇÃO FINAL E REMOÇÃO DE RUÍDOS
# ==========================================
# Remover os casos Censurados e Indeterminados para o treinamento do modelo
df_final = df_transformado[df_transformado['alvo_amamentacao'].isin(['Sucesso (AME 6m+)', 'Desmame precoce (< 6m)'])]

print("Novas features criadas com sucesso!")
print(f"Instâncias finais prontas para o modelo: {len(df_final)}")

# Salvar o dataset pronto
df_final.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')