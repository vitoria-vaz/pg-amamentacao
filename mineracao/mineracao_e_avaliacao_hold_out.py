import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix)

# 1. Mapeamento dos diretórios gerados no pré-processamento
input_base_dirs = ['selecao_dados/qtde_atributos_10/', 'selecao_dados/qtde_atributos_15/']
output_dir = 'interpretacao/'
os.makedirs(output_dir, exist_ok=True)

# 2. Instanciação dos algoritmos
# Fixamos o random_state para garantir a reprodutibilidade dos experimentos
modelos = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

lista_resultados = []
lista_matrizes_confusao = [] # Nova lista para armazenar os dados das matrizes

# Instanciar LabelEncoder para transformar a classe alvo categórica em binária (0 e 1)
# O encoder mapeará alfabeticamente: 0 = 'desmame_precoce_<_6m', 1 = 'sucesso_ame_6m+'
le = LabelEncoder()

# 3. Iteração sobre as pastas de 10 e 15 atributos
for base_dir in input_base_dirs:
    if not os.path.exists(base_dir):
        print(f"Diretório não encontrado: {base_dir}")
        continue
        
    # Extrai o nome '10' ou '15' da pasta para registrar na tabela
    qtde_atributos = base_dir.strip('/').split('_')[-1]
    
    # 4. Iteração sobre as coleções (treino_e_teste_1, treino_e_teste_2, etc.)
    for colecao in os.listdir(base_dir):
        colecao_dir = os.path.join(base_dir, colecao)
        if not os.path.isdir(colecao_dir):
            continue
            
        # Carregar os arquivos CSV separados
        treino = pd.read_csv(os.path.join(colecao_dir, 'dataset_treino.csv'))
        teste = pd.read_csv(os.path.join(colecao_dir, 'dataset_teste.csv'))
        
        # Separar atributos (X) do alvo (y)
        X_treino = treino.drop('alvo_sucesso_ame_6m', axis=1)
        y_treino = treino['alvo_sucesso_ame_6m']
        
        X_teste = teste.drop('alvo_sucesso_ame_6m', axis=1)
        y_teste = teste['alvo_sucesso_ame_6m']

        # --- Tratamento de Atributos Categóricos (Binarização / One-Hot Encoding) ---
        X_treino_encoded = pd.get_dummies(X_treino)
        X_teste_encoded = pd.get_dummies(X_teste)
        
        # Define os caracteres que dão problema e substitui por underline (_) para o XGBoost
        caracteres_proibidos = r'[><=\+\[\]]'

        X_treino_encoded.columns = X_treino_encoded.columns.str.replace(caracteres_proibidos, '_', regex=True)
        X_teste_encoded.columns = X_teste_encoded.columns.str.replace(caracteres_proibidos, '_', regex=True)
        
        # Alinha as colunas de treino e teste
        X_treino_encoded, X_teste_encoded = X_treino_encoded.align(
            X_teste_encoded, join='left', axis=1, fill_value=0
        )
        
        # Codificação Numérica da Variável Alvo 
        y_treino_enc = le.fit_transform(y_treino)
        y_teste_enc = le.transform(y_teste)
        
        # 5. Treinamento e Avaliação para cada modelo
        for nome_modelo, modelo in modelos.items():
            
            # --- Etapa de Mineração (Treinamento) ---
            modelo.fit(X_treino_encoded, y_treino_enc)
            
            # --- Etapa de Interpretação (Predição e Métricas) ---
            y_pred = modelo.predict(X_teste_encoded)
            y_prob = modelo.predict_proba(X_teste_encoded)[:, 1]
            
            # Cálculo das Métricas Diretas
            acc = accuracy_score(y_teste_enc, y_pred)
            sensibilidade = recall_score(y_teste_enc, y_pred) 
            f1 = f1_score(y_teste_enc, y_pred)
            auc_roc = roc_auc_score(y_teste_enc, y_prob)
            
            # Extração dos valores da Matriz de Confusão
            tn, fp, fn, tp = confusion_matrix(y_teste_enc, y_pred).ravel()
            especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # 1. Armazenar as métricas na lista principal
            lista_resultados.append({
                'Quantidade Atributos': qtde_atributos,
                'Coleção': colecao,
                'Algoritmo': nome_modelo,
                'Acurácia': round(acc, 4),
                'Sensibilidade': round(sensibilidade, 4),
                'Especificidade': round(especificidade, 4),
                'F1-Score': round(f1, 4),
                'AUC-ROC': round(auc_roc, 4)
            })

            # 2. Armazenar as matrizes de confusão na nova lista (com nomes explicativos)
            lista_matrizes_confusao.append({
                'Quantidade Atributos': qtde_atributos,
                'Coleção': colecao,
                'Algoritmo': nome_modelo,
                'Verdadeiros Negativos (VN) [Real: Desmame, Previsto: Desmame]': tn,
                'Falsos Positivos (FP) [Real: Desmame, Previsto: Sucesso]': fp,
                'Falsos Negativos (FN) [Real: Sucesso, Previsto: Desmame]': fn,
                'Verdadeiros Positivos (VP) [Real: Sucesso, Previsto: Sucesso]': tp
            })

# 6. Compilação e Exportação
df_resultados = pd.DataFrame(lista_resultados)
df_matrizes = pd.DataFrame(lista_matrizes_confusao)

# Ordenar as tabelas para facilitar a leitura no documento
df_resultados = df_resultados.sort_values(by=['Quantidade Atributos', 'Coleção', 'Algoritmo'])
df_matrizes = df_matrizes.sort_values(by=['Quantidade Atributos', 'Coleção', 'Algoritmo'])

# Caminho do arquivo de saída
caminho_tabela = os.path.join(output_dir, 'tabela_comparativa_algoritmos.xlsx')

# Utilizar o ExcelWriter para exportar múltiplas planilhas
with pd.ExcelWriter(caminho_tabela, engine='openpyxl') as writer:
    df_resultados.to_excel(writer, sheet_name='Métricas', index=False)
    df_matrizes.to_excel(writer, sheet_name='Matrizes de Confusão', index=False)

print(f"Experimentos finalizados com sucesso! Tabela com múltiplas planilhas gerada em: {caminho_tabela}")