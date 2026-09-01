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
from sklearn.preprocessing import LabelEncoder

# Importando ferramentas para codificação e seleção
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif # ADICIONADO: SelectKBest e f_classif

# Importando métricas
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

# ==========================================
# 1. CARREGAR E PREPARAR OS DADOS
# ==========================================
df = pd.read_csv('dataset/dataset_pos_processamento_2.csv')

# Separar todos os atributos (X) e a variável alvo (y)
X = df.drop('alvo_sucesso_ame_6m', axis=1)

# Garantir que a variável alvo seja numérica (0 e 1)
le = LabelEncoder()
y = pd.Series(le.fit_transform(df['alvo_sucesso_ame_6m']))

print("Mapeamento da Variável Alvo:", dict(zip(le.classes_, le.transform(le.classes_))))

# ==========================================
# 2. CONFIGURAÇÕES DOS MODELOS E VALIDAÇÃO
# ==========================================
modelos = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

lista_resultados = []
lista_matrizes_confusao = []

# Lista com as variações da quantidade de atributos que a orientadora sugeriu
lista_k_atributos = [5, 10, 15, 20, 25]

# Identificar quais colunas são categóricas nativamente
colunas_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Cria o transformador para as colunas categóricas
pre_processador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), colunas_categoricas)
    ],
    remainder='passthrough' # Mantém as numéricas inalteradas
)

# ==========================================
# 3. FUNÇÃO PRINCIPAL DE AVALIAÇÃO (ADAPTADA)
# ==========================================
def avaliar_k_atributos(X, y, lista_k):
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
            for train_index, test_index in skf.split(X, y):
                X_treino, X_teste = X.iloc[train_index], X.iloc[test_index]
                y_treino, y_teste = y.iloc[train_index], y.iloc[test_index]
                
                pipeline.fit(X_treino, y_treino)
                
                predicoes = pipeline.predict(X_teste)
                
                # Tratamento para algoritmos que não suportam predict_proba nativamente (se houver no futuro)
                if hasattr(pipeline, "predict_proba"):
                    probabilidades = pipeline.predict_proba(X_teste)[:, 1] 
                else:
                    probabilidades = predicoes
                    
                y_verdadeiros.extend(y_teste)
                y_preditos.extend(predicoes)
                y_probabilidades.extend(probabilidades)
            
            # -- Calculando Métricas Globais --
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

# Criando o DataFrame Consolidado (Média e Desvio Padrão)
colunas_metricas = ['Acurácia', 'Sensibilidade', 'Especificidade', 'F1-Score', 'AUC-ROC']

# ALTERAÇÃO: Agrupando APENAS por Algoritmo (calcula a média e std de todos os K para cada modelo)
df_consolidado = df_resultados.groupby(['Algoritmo'])[colunas_metricas].agg(['mean', 'std']).reset_index()

# Achatando os nomes das colunas multinível
df_consolidado.columns = ['_'.join(col).strip('_') if type(col) is tuple else col for col in df_consolidado.columns.values]
df_consolidado = df_consolidado.round(4)

# Salvar as tabelas
os.makedirs('interpretacao', exist_ok=True)
nome_arquivo_excel = 'interpretacao/resultados_modelos_10fold_SelectKBest.xlsx'

with pd.ExcelWriter(nome_arquivo_excel) as writer:
    df_resultados.to_excel(writer, sheet_name='Métricas', index=False)
    df_matrizes.to_excel(writer, sheet_name='Matrizes de Confusão', index=False)
    df_consolidado.to_excel(writer, sheet_name='Consolidado por Algoritmo', index=False)

print(f"\nArquivo Excel '{nome_arquivo_excel}' gerado com sucesso!")