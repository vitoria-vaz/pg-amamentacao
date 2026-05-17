import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_discretizado.csv'
CAMINHO_SAIDA   = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_pronto.csv'

df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

# ==========================================
# 2. TRATAMENTO CRÍTICO (Remoção da base de predição)
# ==========================================
# Precisamos dropar os nulos da idade da mãe (7 casos) para o modelo de ML não quebrar
df = df.dropna(subset=['idade_mae_cat'])
print(f"Instâncias com idade nula removidas. Linhas restantes: {len(df)}")

# PREENCHIMENTO DOS NULOS GERADOS PELA DISCRETIZAÇÃO
# Como o código 99 (Ignorado) virou NaN no corte, preenchemos com a moda
if 'gestacoes_cat' in df.columns:
    moda_gest = df['gestacoes_cat'].mode()[0]
    df['gestacoes_cat'] = df['gestacoes_cat'].fillna(moda_gest)
    
if 'filhos_vivos_cat' in df.columns:
    moda_filhos = df['filhos_vivos_cat'].mode()[0]
    df['filhos_vivos_cat'] = df['filhos_vivos_cat'].fillna(moda_filhos)

print("✅ Valores nulos de Gestações e Filhos Vivos preenchidos com a Moda.")

# ==========================================
# 3. IMPUTAÇÃO PREDITIVA (Decision Tree)
# ==========================================
print("Iniciando Imputação Preditiva para nulos do Pré-Natal...")

# --- A. Prever 'inic_prenat' ---
preditores_inic = ['q07_renda_faixa', 'a00_regiao', 'idade_mae_cat']
encoder_inic = OrdinalEncoder()

df_encoded = df.copy()
df_encoded[preditores_inic] = encoder_inic.fit_transform(df[preditores_inic].astype(str))

treino_inic = df_encoded[df_encoded['inic_prenat'].notna()]
pred_inic = df_encoded[df_encoded['inic_prenat'].isna()]

if len(pred_inic) > 0:
    clf_inic = DecisionTreeClassifier(random_state=42)
    clf_inic.fit(treino_inic[preditores_inic], treino_inic['inic_prenat'])
    df.loc[df['inic_prenat'].isna(), 'inic_prenat'] = clf_inic.predict(pred_inic[preditores_inic])
print("✅ Início do Pré-natal preenchido com sucesso!")

# --- B. Prever 'num_consultas' ---
preditores_num = ['q07_renda_faixa', 'a00_regiao', 'inic_prenat']
encoder_num = OrdinalEncoder()

df_encoded_num = df.copy()
df_encoded_num[preditores_num] = encoder_num.fit_transform(df[preditores_num].astype(str))

treino_num = df_encoded_num[df_encoded_num['num_consultas'].notna()]
pred_num = df_encoded_num[df_encoded_num['num_consultas'].isna()]

if len(pred_num) > 0:
    clf_num = DecisionTreeClassifier(random_state=42)
    clf_num.fit(treino_num[preditores_num], treino_num['num_consultas'])
    df.loc[df['num_consultas'].isna(), 'num_consultas'] = clf_num.predict(pred_num[preditores_num])
print("✅ Número de Consultas preenchido com sucesso!")

# ==========================================
# 4. VERIFICAÇÃO FINAL E SALVAMENTO
# ==========================================
total_nulos_restantes = df.isnull().sum().sum()
if total_nulos_restantes == 0:
    df.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
    print(f"\n🚀 Dataset PRONTO PARA A REDE BAYESIANA (0 nulos) salvo em:\n{CAMINHO_SAIDA}")
else:
    print(f"\n⚠️ ATENÇÃO: Ainda existem {total_nulos_restantes} valores nulos no dataset.")