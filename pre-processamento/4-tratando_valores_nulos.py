import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_discretizado.csv'
CAMINHO_SAIDA   = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_pronto.csv'

df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')

# ==========================================
# 2. TRATAMENTO CRÍTICO E MODA
# ==========================================
df = df.dropna(subset=['idade_mae_cat'])
print(f"Instâncias com idade nula removidas. Linhas restantes: {len(df)}")

if 'gestacoes_cat' in df.columns:
    moda_gest = df['gestacoes_cat'].mode()[0]
    df['gestacoes_cat'] = df['gestacoes_cat'].fillna(moda_gest)
    
if 'filhos_vivos_cat' in df.columns:
    moda_filhos = df['filhos_vivos_cat'].mode()[0]
    df['filhos_vivos_cat'] = df['filhos_vivos_cat'].fillna(moda_filhos)

print("Valores nulos de Gestações e Filhos Vivos preenchidos com a Moda.")

# ==========================================
# 3. IMPUTAÇÃO PREDITIVA TRANSPARENTE (Decision Tree)
# ==========================================
print("\n" + "="*50)
print("INICIANDO IMPUTAÇÃO PREDITIVA COM AVALIAÇÃO (HOLDOUT)")
print("="*50)

# ---------------------------------------------------------
# --- A. Prever 'inic_prenat' ---
# ---------------------------------------------------------
print("\n>>> ETAPA A: Previsão do 'Início do Pré-natal'")
preditores_inic = ['q07_renda_faixa', 'a00_regiao', 'idade_mae_cat']
encoder_inic = OrdinalEncoder()

df_encoded = df.copy()
df_encoded[preditores_inic] = encoder_inic.fit_transform(df[preditores_inic].astype(str))

# Separa quem tem gabarito (para treino/teste) e quem não tem (nulos)
dados_completos = df_encoded[df_encoded['inic_prenat'].notna()]
dados_nulos = df_encoded[df_encoded['inic_prenat'].isna()]

if len(dados_nulos) > 0:
    X_inic = dados_completos[preditores_inic]
    y_inic = dados_completos['inic_prenat']

    # 1. Aplicando Holdout (80% Treino / 20% Teste)
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_inic, y_inic, test_size=0.2, random_state=42)

    # 2. Treinando a Árvore apenas no Treino
    clf_teste_inic = DecisionTreeClassifier(random_state=42)
    clf_teste_inic.fit(X_train_i, y_train_i)

    # 3. Medindo a Acurácia no Teste
    previsoes_teste_i = clf_teste_inic.predict(X_test_i)
    acuracia_inic = accuracy_score(y_test_i, previsoes_teste_i)
    print(f"📊 Acurácia do modelo Holdout: {acuracia_inic * 100:.2f}%")

    # 4. Retreinando com 100% dos dados completos para imputação final (Maior precisão)
    clf_final_inic = DecisionTreeClassifier(random_state=42)
    clf_final_inic.fit(X_inic, y_inic)

    # 5. Imputando os valores nos espaços em branco
    df.loc[df['inic_prenat'].isna(), 'inic_prenat'] = clf_final_inic.predict(dados_nulos[preditores_inic])
    print("✅ Nulos de 'inic_prenat' preenchidos!")

# ---------------------------------------------------------
# --- B. Prever 'num_consultas' ---
# ---------------------------------------------------------
print("\n>>> ETAPA B: Previsão do 'Número de Consultas' (Encadeada)")
preditores_num = ['q07_renda_faixa', 'a00_regiao', 'inic_prenat']
encoder_num = OrdinalEncoder()

df_encoded_num = df.copy()
df_encoded_num[preditores_num] = encoder_num.fit_transform(df[preditores_num].astype(str))

# Separa quem tem gabarito e quem não tem
dados_completos_num = df_encoded_num[df_encoded_num['num_consultas'].notna()]
dados_nulos_num = df_encoded_num[df_encoded_num['num_consultas'].isna()]

if len(dados_nulos_num) > 0:
    X_num = dados_completos_num[preditores_num]
    y_num = dados_completos_num['num_consultas']

    # 1. Aplicando Holdout (80% Treino / 20% Teste)
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_num, y_num, test_size=0.2, random_state=42)

    # 2. Treinando a Árvore apenas no Treino
    clf_teste_num = DecisionTreeClassifier(random_state=42)
    clf_teste_num.fit(X_train_n, y_train_n)

    # 3. Medindo a Acurácia no Teste
    previsoes_teste_n = clf_teste_num.predict(X_test_n)
    acuracia_num = accuracy_score(y_test_n, previsoes_teste_n)
    print(f"📊 Acurácia do modelo Holdout: {acuracia_num * 100:.2f}%")

    # 4. Retreinando com 100% dos dados completos
    clf_final_num = DecisionTreeClassifier(random_state=42)
    clf_final_num.fit(X_num, y_num)

    # 5. Imputando os valores nos espaços em branco
    df.loc[df['num_consultas'].isna(), 'num_consultas'] = clf_final_num.predict(dados_nulos_num[preditores_num])
    print("✅ Nulos de 'num_consultas' preenchidos!")

# ==========================================
# 4. VERIFICAÇÃO FINAL E SALVAMENTO
# ==========================================
total_nulos_restantes = df.isnull().sum().sum()
print("\n" + "="*50)
if total_nulos_restantes == 0:
    df.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
    print(f"🚀 Dataset PRONTO PARA A REDE BAYESIANA (0 nulos) salvo em:\n{CAMINHO_SAIDA}")
else:
    print(f"⚠️ ATENÇÃO: Ainda existem {total_nulos_restantes} valores nulos no dataset.")
print("="*50)