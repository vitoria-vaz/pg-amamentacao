import pandas as pd
import numpy as np
import os
import warnings

# Suprimir avisos para manter o console limpo durante o Cross-Validation
warnings.filterwarnings('ignore')

# Importando bibliotecas de Machine Learning
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression  # NOVO: Regressão Logística
from sklearn.preprocessing import LabelEncoder

# Importando ferramentas para codificação e seleção
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # NOVO: StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif # ADICIONADO: SelectKBest e f_classif

# Importando métricas
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\dataset_final_critico.csv'
ALVO = 'aleitamento_materno_exclusivo'
# ==========================================
# 1. CARREGAR E PREPARAR OS DADOS
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA)

# Separar todos os atributos (X) e a variável alvo (y)
X = df.drop(ALVO, axis=1)

# Garantir que a variável alvo seja numérica (0 e 1)
le = LabelEncoder()
y = pd.Series(le.fit_transform(df[ALVO]))

print("Mapeamento da Variável Alvo:", dict(zip(le.classes_, le.transform(le.classes_))))

# ==========================================
# 2. CONFIGURAÇÕES DOS MODELOS E VALIDAÇÃO
# ==========================================
modelos = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)  # NOVO
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

lista_resultados = []
lista_matrizes_confusao = []
lista_atributos_selecionados = []
lista_resultados_por_fold = []

# Lista com as variações da quantidade de atributos que a orientadora sugeriu
lista_k_atributos = [5, 10, 15, 19]

# Identificar quais colunas são categóricas nativamente
colunas_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Cria o transformador para as colunas categóricas e numéricas
# NOVO: StandardScaler adicionado nas colunas numéricas (remainder), pois a Regressão
# Logística é sensível à escala dos atributos - as demais árvores/ensemble não são afetadas por isso.
pre_processador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), colunas_categoricas),
        ('num', StandardScaler(), [col for col in X.columns if col not in colunas_categoricas])  # NOVO
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

# ==========================================
# 3. FUNÇÃO PRINCIPAL DE AVALIAÇÃO (ADAPTADA)
# ==========================================
def avaliar_k_atributos(X, y, lista_k):
    primeiro_modelo = list(modelos.keys())[0]

    for k in lista_k:
        print(f"\nAvaliando o impacto de selecionar os {k} melhores atributos...")

        for nome_modelo, modelo in modelos.items():

            # Cria a Linha de Montagem (Pipeline)
            # O SelectKBest entra AGORA, após o pré-processador transformar os dados categóricos
            pipeline = Pipeline(steps=[
                ('pre_processador', pre_processador),
                ('seletor_features', SelectKBest(score_func=f_classif, k=k)),
                ('modelo', modelo)
            ])

            y_verdadeiros = []
            y_preditos = []
            y_probabilidades = []

            # Executa o 10-fold CV
            for numero_fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
                X_treino, X_teste = X.iloc[train_index], X.iloc[test_index]
                y_treino, y_teste = y.iloc[train_index], y.iloc[test_index]

                pipeline.fit(X_treino, y_treino)

                # Captura os atributos selecionados (independe do modelo, só do k e do fold)
                if nome_modelo == primeiro_modelo:
                    nomes_pos_preprocessamento = pipeline.named_steps['pre_processador'].get_feature_names_out()
                    mascara_selecao = pipeline.named_steps['seletor_features'].get_support()
                    atributos_do_fold = nomes_pos_preprocessamento[mascara_selecao]
                    for atributo in atributos_do_fold:
                        lista_atributos_selecionados.append({
                            'Quantidade Atributos (K)': k,
                            'Fold': numero_fold,
                            'Atributo': atributo
                        })

                predicoes = pipeline.predict(X_teste)

                # Tratamento para algoritmos que não suportam predict_proba nativamente (se houver no futuro)
                if hasattr(pipeline, "predict_proba"):
                    probabilidades = pipeline.predict_proba(X_teste)[:, 1]
                else:
                    probabilidades = predicoes

                # -- Métricas SÓ deste fold (DENTRO do for numero_fold) --
                cm_fold = confusion_matrix(y_teste, predicoes)
                tn_f, fp_f, fn_f, tp_f = cm_fold.ravel()
                lista_resultados_por_fold.append({
                    'Quantidade Atributos (K)': k,
                    'Algoritmo': nome_modelo,
                    'Fold': numero_fold,
                    'Acurácia': round(accuracy_score(y_teste, predicoes), 4),
                    'F1-Score': round(f1_score(y_teste, predicoes), 4),
                    'AUC-ROC': round(roc_auc_score(y_teste, probabilidades), 4)
                })

                y_verdadeiros.extend(y_teste)
                y_preditos.extend(predicoes)
                y_probabilidades.extend(probabilidades)

            # -- Calculando Métricas Globais (DENTRO do for nome_modelo, FORA do for numero_fold) --
            cm = confusion_matrix(y_verdadeiros, y_preditos)
            tn, fp, fn, tp = cm.ravel()

            acc = accuracy_score(y_verdadeiros, y_preditos)
            sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
            especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = f1_score(y_verdadeiros, y_preditos)
            auc_roc = roc_auc_score(y_verdadeiros, y_probabilidades)

            # Adiciona métricas na lista principal
            lista_resultados.append({
                'Quantidade Atributos (K)': k,
                'Algoritmo': nome_modelo,
                'Acurácia': round(acc, 4),
                'Sensibilidade': round(sensibilidade, 4),
                'Especificidade': round(especificidade, 4),
                'F1-Score': round(f1, 4),
                'AUC-ROC': round(auc_roc, 4)
            })

            # Adiciona as matrizes de confusão
            lista_matrizes_confusao.append({
                'Quantidade Atributos (K)': k,
                'Algoritmo': nome_modelo,
                'Verdadeiros Negativos (VN) [Real: Desmame, Previsto: Desmame]': tn,
                'Falsos Positivos (FP) [Real: Desmame, Previsto: Sucesso]': fp,
                'Falsos Negativos (FN) [Real: Sucesso, Previsto: Desmame]': fn,
                'Verdadeiros Positivos (VP) [Real: Sucesso, Previsto: Sucesso]': tp
            })

# ==========================================
# 4. EXECUÇÃO E EXPORTAÇÃO
# ==========================================
# Executa a função passando o X completo e a lista de K desejada
avaliar_k_atributos(X, y, lista_k_atributos)

print("\nProcessamento dos modelos concluído! Gerando Excel...")

df_resultados = pd.DataFrame(lista_resultados)
df_matrizes = pd.DataFrame(lista_matrizes_confusao)
df_atributos = pd.DataFrame(lista_atributos_selecionados)
df_resultados_por_fold = pd.DataFrame(lista_resultados_por_fold)

# Verificação rápida de sanidade: agora com 5 modelos
# 4 valores de K x 5 modelos = 20 | 4 valores de K x 5 modelos x 10 folds = 200
print(f"Total de linhas em df_resultados (esperado 20): {len(df_resultados)}")
print(f"Total de linhas em df_resultados_por_fold (esperado 200): {len(df_resultados_por_fold)}")

# Conta em quantos dos 10 folds cada atributo foi selecionado, por K
df_atributos_resumo = (
    df_atributos
    .groupby(['Quantidade Atributos (K)', 'Atributo'])
    .size()
    .reset_index(name='Vezes Selecionado (de 10 folds)')
    .sort_values(
        ['Quantidade Atributos (K)', 'Vezes Selecionado (de 10 folds)'],
        ascending=[True, False]
    )
)

df_atributos_concatenados = (
    df_atributos
    .groupby(['Quantidade Atributos (K)', 'Fold'])['Atributo']
    .apply(lambda atributos: ', '.join(atributos))
    .reset_index(name='Atributos Selecionados')
)

# Junta métrica por fold + lista de atributos daquele fold
df_analise_fold = df_resultados_por_fold.merge(
    df_atributos_concatenados,
    on=['Quantidade Atributos (K)', 'Fold'],
    how='left'
)

# Criando o DataFrame Consolidado (Média e Desvio Padrão)
colunas_metricas = ['Acurácia', 'Sensibilidade', 'Especificidade', 'F1-Score', 'AUC-ROC']

# Agrupando APENAS por Algoritmo (calcula a média e std de todos os K para cada modelo)
df_consolidado = df_resultados.groupby(['Algoritmo'])[colunas_metricas].agg(['mean', 'std']).reset_index()

# Achatando os nomes das colunas multinível
df_consolidado.columns = ['_'.join(col).strip('_') if type(col) is tuple else col for col in df_consolidado.columns.values]
df_consolidado = df_consolidado.round(4)

# Salvar as tabelas
nome_arquivo_excel = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\interpretacao\testando_artigo_concorrente\resultados_19_atributos_10fold_SelectKBest.xlsx'
os.makedirs(os.path.dirname(nome_arquivo_excel), exist_ok=True)

with pd.ExcelWriter(nome_arquivo_excel) as writer:
    df_resultados.to_excel(writer, sheet_name='Métricas', index=False)
    df_matrizes.to_excel(writer, sheet_name='Matrizes de Confusão', index=False)
    df_consolidado.to_excel(writer, sheet_name='Consolidado por Algoritmo', index=False)
    df_atributos_resumo.to_excel(writer, sheet_name='Atributos Selecionados', index=False)
    df_analise_fold.to_excel(writer, sheet_name='Métrica x Atributos por Fold', index=False)

print(f"\nArquivo Excel '{nome_arquivo_excel}' gerado com sucesso!")