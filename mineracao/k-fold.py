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

# NOVOS IMPORTS: Ferramentas para codificação mais eficiente
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# ==========================================
# 2. DICIONÁRIOS DE ATRIBUTOS
# ==========================================
colecao_10_atributos = {
    "treino_e_teste_1" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "recebeu_outro_leite", "usou_bico_artificial", "nivel_inseguranca_alimentar", "usou_bomba_extracao", "inicio_prenatal", "reside_com_parceiro", "zona_residencial"],
    "treino_e_teste_2" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "usou_bico_artificial", "recebeu_outro_leite", "zona_residencial", "usou_sondinha_relactacao", "faixa_etaria_mae", "usou_bomba_extracao", "regiao_residencia"],
    "treino_e_teste_3" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "recebeu_outro_leite", "faixa_etaria_mae", "regiao_residencia", "usou_bico_artificial", "nivel_inseguranca_alimentar", "faixa_renda_familiar", "escolaridade_mae"],
    "treino_e_teste_4" : ["historico_uso_chupeta", "faixa_renda_familiar", "busca_informacao_aleitamento", "classificacao_peso_nascimento", "zona_residencial", "regiao_residencia", "usou_sondinha_relactacao", "usou_copinho", "qtd_filhos_vivos", "tipo_parto"],
    "treino_e_teste_5" : ["historico_uso_chupeta", "regiao_residencia", "usou_mamadeira", "busca_informacao_aleitamento", "oferta_outros_liquidos", "recebeu_outro_leite", "escolaridade_mae", "nivel_inseguranca_alimentar", "inicio_prenatal", "recebe_auxilio_governamental"]
}

colecao_15_atributos = {
    "treino_e_teste_1" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "recebeu_outro_leite", "usou_bico_artificial", "nivel_inseguranca_alimentar", "usou_bomba_extracao", "inicio_prenatal", "reside_com_parceiro", "zona_residencial", "regiao_residencia", "recebe_auxilio_governamental", "busca_informacao_aleitamento", "tempo_ate_primeira_mamada", "faixa_etaria_mae"],
    "treino_e_teste_2" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "usou_bico_artificial", "recebeu_outro_leite", "zona_residencial", "usou_sondinha_relactacao", "faixa_etaria_mae", "usou_bomba_extracao", "regiao_residencia", "nivel_inseguranca_alimentar", "inicio_prenatal", "escolaridade_mae", "tempo_ate_primeira_mamada", "faixa_consultas_prenatal"],
    "treino_e_teste_3" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "recebeu_outro_leite", "faixa_etaria_mae", "regiao_residencia", "usou_bico_artificial", "nivel_inseguranca_alimentar", "faixa_renda_familiar", "escolaridade_mae", "usou_bomba_extracao", "busca_informacao_aleitamento", "tempo_ate_primeira_mamada", "raca_cor_mae", "inicio_prenatal"],
    "treino_e_teste_4" : ["historico_uso_chupeta", "faixa_renda_familiar", "busca_informacao_aleitamento", "classificacao_peso_nascimento", "zona_residencial", "regiao_residencia", "usou_sondinha_relactacao", "usou_copinho", "qtd_filhos_vivos", "tipo_parto", "usou_concha_amamentacao", "oferta_outros_liquidos", "usou_protetor_mamilo", "usou_bomba_extracao", "usou_bico_artificial"],
    "treino_e_teste_5" : ["historico_uso_chupeta", "regiao_residencia", "usou_mamadeira", "busca_informacao_aleitamento", "oferta_outros_liquidos", "recebeu_outro_leite", "escolaridade_mae", "nivel_inseguranca_alimentar", "inicio_prenatal", "recebe_auxilio_governamental", "raca_cor_mae", "usou_bico_artificial", "usou_bomba_extracao", "adequacao_peso_idade_gestacional", "faixa_renda_familiar"]
}

colecao_todos_atributos = {
    "treino_e_teste_todos" : ["recebeu_outro_leite", "oferta_outros_liquidos", "usou_concha_amamentacao", "usou_protetor_mamilo", "usou_bico_artificial",
                              "usou_bomba_extracao", "usou_mamadeira", "usou_sondinha_relactacao", "usou_copinho", "busca_informacao_aleitamento",
                              "regiao_residencia", "zona_residencial", "tipo_parto", "historico_uso_chupeta", "reside_com_parceiro", "situacao_laboral_mae", 
                              "realizou_prenatal", "recebe_auxilio_governamental", "faixa_renda_familiar", "nivel_inseguranca_alimentar", "tempo_ate_primeira_mamada", 
                              "classificacao_peso_nascimento", "inicio_prenatal", "faixa_consultas_prenatal", "adequacao_peso_idade_gestacional", 
                              "escolaridade_mae", "raca_cor_mae", "faixa_etaria_mae", "qtd_filhos_vivos"] 
} 

# ==========================================
# 3. CONFIGURAÇÕES DOS MODELOS
# ==========================================
modelos = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss') # Removi use_label_encoder (obsoleto em versões recentes)
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

lista_resultados = []
lista_matrizes_confusao = []

# ==========================================
# 4. FUNÇÃO PRINCIPAL DE AVALIAÇÃO
# ==========================================
def avaliar_colecoes(dict_colecoes, qtde_atributos):
    for nome_colecao, atributos in dict_colecoes.items():
        print(f"Avaliando coleção: {nome_colecao} ({qtde_atributos} atributos originais)...")
        
        # Filtra X apenas com os atributos da coleção atual
        X_subset = X[atributos]
        
        # ADAPTAÇÃO: Identificar quais colunas são categóricas e quais são numéricas
        # Isto previne que o codificador tente transformar números em categorias
        colunas_categoricas = X_subset.select_dtypes(include=['object', 'category']).columns.tolist()
        colunas_numericas = X_subset.select_dtypes(exclude=['object', 'category']).columns.tolist()
        
        # Cria o transformador para as colunas categóricas
        # handle_unknown='ignore' garante que se aparecer uma categoria nova no teste, não haverá erro
        pre_processador = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), colunas_categoricas)
            ],
            remainder='passthrough' # Mantém as numéricas inalteradas
        )
        
        for nome_modelo, modelo in modelos.items():
            
            # Cria a Linha de Montagem (Pipeline)
            # 1º Passo: Aplica o One-Hot Encoding (pre_processador)
            # 2º Passo: Aplica o Modelo de Machine Learning
            pipeline = Pipeline(steps=[
                ('pre_processador', pre_processador),
                ('modelo', modelo)
            ])
            
            y_verdadeiros = []
            y_preditos = []
            y_probabilidades = []
            
            # Executa o 10-fold CV usando a base original (o pipeline faz as transformações!)
            for train_index, test_index in skf.split(X_subset, y):
                X_treino, X_teste = X_subset.iloc[train_index], X_subset.iloc[test_index]
                y_treino, y_teste = y.iloc[train_index], y.iloc[test_index]
                
                # O pipeline transforma os dados de treino e depois treina o modelo
                pipeline.fit(X_treino, y_treino)
                
                # O pipeline transforma os dados de teste (usando o padrão do treino) e faz as predições
                predicoes = pipeline.predict(X_teste)
                probabilidades = pipeline.predict_proba(X_teste)[:, 1] 
                
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
                'Quantidade Atributos': qtde_atributos,
                'Coleção': nome_colecao,
                'Algoritmo': nome_modelo,
                'Acurácia': round(acc, 4),
                'Sensibilidade': round(sensibilidade, 4),
                'Especificidade': round(especificidade, 4),
                'F1-Score': round(f1, 4),
                'AUC-ROC': round(auc_roc, 4)
            })

            # Adiciona as matrizes de confusão
            lista_matrizes_confusao.append({
                'Quantidade Atributos': qtde_atributos,
                'Coleção': nome_colecao,
                'Algoritmo': nome_modelo,
                'Verdadeiros Negativos (VN)': tn,
                'Falsos Positivos (FP)': fp,
                'Falsos Negativos (FN)': fn,
                'Verdadeiros Positivos (VP)': tp
            })

# ==========================================
# 5. EXECUÇÃO E EXPORTAÇÃO
# ==========================================
avaliar_colecoes(colecao_10_atributos, 10)
avaliar_colecoes(colecao_15_atributos, 15)
avaliar_colecoes(colecao_todos_atributos, 29)

print("\nProcessamento dos modelos concluído! Gerando Excel...")

df_resultados = pd.DataFrame(lista_resultados)
df_matrizes = pd.DataFrame(lista_matrizes_confusao)

# Salvar as tabelas
# Criei a pasta interpretacao se ela não existir
os.makedirs('interpretacao', exist_ok=True)
nome_arquivo_excel = 'interpretacao/resultados_modelos_10fold.xlsx'

with pd.ExcelWriter(nome_arquivo_excel) as writer:
    df_resultados.to_excel(writer, sheet_name='Métricas', index=False)
    df_matrizes.to_excel(writer, sheet_name='Matrizes de Confusão', index=False)

print(f"\nArquivo Excel '{nome_arquivo_excel}' gerado com sucesso!")