import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

# 1. Carregar os dados limpos
# Presume-se que o dataset aqui já possua os 30 atributos + a variável alvo
df = pd.read_csv('dataset/dataset_pos_processamento.csv')

# 2. Separar todos os atributos (X) e a variável alvo (y)
X = df.drop('alvo_sucesso_ame_6m', axis=1)
y = df['alvo_sucesso_ame_6m']

# 3. Configurar o K-Fold Estratificado para 5 divisões
# shuffle=True garante que os dados sejam embaralhados antes de dividir
# random_state=42 garante a reprodutibilidade
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Diretório base de saída
base_output_dir = 'selecao_dados/todos_30_atributos_kfold/'

# 5. Iterar sobre as 5 divisões geradas pelo StratifiedKFold
# O skf.split retorna os índices das linhas para treino e teste em cada rodada
for fold, (indice_treino, indice_teste) in enumerate(skf.split(X, y), start=1):
    
    # Criar um subdiretório específico para o fold atual
    # Ex: 'selecao_dados/todos_30_atributos_kfold/fold_1/'
    output_dir = os.path.join(base_output_dir, f'fold_{fold}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar as linhas de X e y usando os índices gerados
    X_treino, X_teste = X.iloc[indice_treino], X.iloc[indice_teste]
    y_treino, y_teste = y.iloc[indice_treino], y.iloc[indice_teste]
    
    # Juntar o X com o y correspondente
    # A concatenação usa o índice original das linhas, mantendo a integridade dos dados
    dataset_treino = pd.concat([X_treino, y_treino], axis=1)
    dataset_teste = pd.concat([X_teste, y_teste], axis=1)
    
    # Definir o caminho dos arquivos
    caminho_treino = os.path.join(output_dir, 'dataset_treino.csv')
    caminho_teste = os.path.join(output_dir, 'dataset_teste.csv')
    
    # Exportar para CSV sem a coluna de índice do Pandas
    dataset_treino.to_csv(caminho_treino, index=False)
    dataset_teste.to_csv(caminho_teste, index=False)
    
    print(f"Arquivos gerados com sucesso para o Fold {fold}")

print("\nProcessamento concluído: Os 5 conjuntos (folds) com todos os 30 atributos foram salvos!")