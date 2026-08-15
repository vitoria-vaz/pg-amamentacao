import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 1. Importações da pgmpy ajustadas
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFWriter

# Tratamento para as diferenças de versão da pgmpy:
try:
    from pgmpy.causal_discovery import HillClimbSearch
    IS_NEW_API = True
except ImportError:
    from pgmpy.estimators import HillClimbSearch
    IS_NEW_API = False

# 2. Mapeamento do diretório gerado no pré-processamento via K-Fold
base_dir = 'selecao_dados/todos_30_atributos_kfold/'
output_dir = 'interpretacao/'
os.makedirs(output_dir, exist_ok=True)

lista_resultados = []
lista_matrizes_confusao = [] 

# Instanciar LabelEncoder para avaliação de métricas
le = LabelEncoder()

# 3. Verificação do diretório
if not os.path.exists(base_dir):
    print(f"Diretório não encontrado: {base_dir}")
else:
    folds = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    # 4. Iteração sobre os 5 Folds
    for fold in folds:
        fold_dir = os.path.join(base_dir, fold)
        
        # Carregar os arquivos CSV originais (SEM get_dummies para a pgmpy)
        treino = pd.read_csv(os.path.join(fold_dir, 'dataset_treino.csv'))
        teste = pd.read_csv(os.path.join(fold_dir, 'dataset_teste.csv'))
        
        target_col = 'alvo_sucesso_ame_6m'
        
        # --- Etapa de Mineração (Treinamento da Rede Bayesiana) ---
        print(f"Aprendendo a estrutura e parâmetros para o {fold}...")
        
        # A. Aprendizado de Estrutura
        if IS_NEW_API:
            # Na nova API (v1.3.0+), instanciamos a classe com a métrica
            hc = HillClimbSearch(scoring_method='bic-d')
            # Executamos o treinamento passando os dados
            hc.fit(treino)
            # O grafo resultante fica armazenado no atributo causal_graph_
            best_model = hc.causal_graph_
        else:
            # Comportamento da API antiga
            hc = HillClimbSearch(treino)
            best_model = hc.estimate(scoring_method='bic-d')
        
        # Criar a Rede Bayesiana com as arestas descobertas
        model = BayesianNetwork(best_model.edges())
        
        # B. Aprendizado de Parâmetros (Estimativa de Máxima Verossimilhança)
        model.fit(treino, estimator=MaximumLikelihoodEstimator)
        
        # --- Etapa de Exportação para o software GeNIe ---
        bif_file = os.path.join(output_dir, f'rede_bayesiana_30_attr_{fold}.bif')
        writer = BIFWriter(model)
        writer.write_bif(bif_file)
        print(f"Grafo do {fold} exportado para {bif_file}")

        # --- Etapa de Interpretação (Predição e Métricas) ---
        X_teste = teste.drop(target_col, axis=1)
        y_teste = teste[target_col]
        
        # Codificação Numérica apenas para calcular as métricas do sklearn
        le.fit(y_teste)
        y_teste_enc = le.transform(y_teste)
        
        # Previsão da classe (Retorna um DataFrame com a coluna alvo)
        y_pred_df = model.predict(X_teste)
        y_pred = le.transform(y_pred_df[target_col])
        
        # Previsão das probabilidades
        y_prob_df = model.predict_probability(X_teste)
        
        # Identificar qual coluna do DataFrame corresponde à classe positiva
        estado_positivo = le.inverse_transform([1])[0] 
        coluna_prob_positiva = f"{target_col}_{estado_positivo}"
        
        if coluna_prob_positiva in y_prob_df.columns:
            y_prob = y_prob_df[coluna_prob_positiva].values
        else:
            y_prob = y_prob_df.iloc[:, 1].values
        
        # Cálculo das Métricas Diretas
        acc = accuracy_score(y_teste_enc, y_pred)
        sensibilidade = recall_score(y_teste_enc, y_pred) 
        f1 = f1_score(y_teste_enc, y_pred)
        auc_roc = roc_auc_score(y_teste_enc, y_prob)
        
        # Extração dos valores da Matriz de Confusão
        tn, fp, fn, tp = confusion_matrix(y_teste_enc, y_pred).ravel()
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Armazenar métricas
        lista_resultados.append({
            'Quantidade Atributos': 30,
            'Fold': fold,
            'Algoritmo': 'Rede Bayesiana',
            'Acurácia': round(acc, 4),
            'Sensibilidade': round(sensibilidade, 4),
            'Especificidade': round(especificidade, 4),
            'F1-Score': round(f1, 4),
            'AUC-ROC': round(auc_roc, 4)
        })

        lista_matrizes_confusao.append({
            'Quantidade Atributos': 30,
            'Fold': fold,
            'Algoritmo': 'Rede Bayesiana',
            'Verdadeiros Negativos (VN)': tn,
            'Falsos Positivos (FP)': fp,
            'Falsos Negativos (FN)': fn,
            'Verdadeiros Positivos (VP)': tp
        })

# 5. Compilação e Exportação para Excel
if lista_resultados:
    df_resultados = pd.DataFrame(lista_resultados)
    df_matrizes = pd.DataFrame(lista_matrizes_confusao)

    caminho_tabela = os.path.join(output_dir, 'tabela_comparativa_redes_bayesianas.xlsx')

    with pd.ExcelWriter(caminho_tabela, engine='openpyxl') as writer:
        df_resultados.to_excel(writer, sheet_name='Métricas', index=False)
        df_matrizes.to_excel(writer, sheet_name='Matrizes de Confusão', index=False)

    print(f"\nTreino concluído! Tabela gerada em: {caminho_tabela}")