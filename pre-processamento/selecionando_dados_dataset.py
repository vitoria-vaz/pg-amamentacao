import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. Carregar os dados limpos
df = pd.read_csv('dataset/dataset_pos_processamento.csv')

# 2. Separar todos os atributos (X) e a variável alvo (y)
X = df.drop('alvo', axis=1)
y = df['alvo']

# 3. Divisão estratificada global (80% treino, 20% teste)
# É fundamental fazer o split ANTES de filtrar as colunas.
# Isso garante que a mesma semente (random_state) mantenha exatamente 
# os mesmos registros nos treinos e testes de todas as coleções.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Dicionário de coleções de atributos
colecao_10_atributos = {
    "treino_e_teste_1" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "recebeu_outro_leite", "usou_bico_artificial", "nivel_inseguranca_alimentar", "usou_bomba_extracao", "inicio_prenatal", "reside_com_parceiro", "zona_residencial"],
    "treino_e_teste_2" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "usou_bico_artificial", "recebeu_outro_leite", "zona_residencial", "usou_sondinha_relactacao", "faixa_etaria_mae", "usou_bomba_extracao", "regiao_residencia"],
    "treino_e_teste_3" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "recebeu_outro_leite", "faixa_etaria_mae", "regiao_residencia", "usou_bico_artificial", "nivel_inseguranca_alimentar", "faixa_renda_familiar", "escolaridade_mae"],
    "treino_e_teste_4" : ["historico_uso_chupeta", "faixa_renda_familiar", "busca_informacao_aleitamento", "classificacao_peso_nascimento", "zona_residencial", "regiao_residencia", "usou_sondinha_relactacao", "usou_copinho", "qtd_filhos_vivos", "tipo_parto"],
    "treino_e_teste_5" : ["historico_uso_chupeta", "regiao_residencia", "usou_mamadeira", "busca_informacao_aleitamento", "oferta_outros_liquidos", "recebeu_outro_leite", "escolaridade_mae", "nivel_inseguranca_alimentar", "inicio_prenatal", "recebe_auxilio_governamental"]
}

colecao_15_atributos = {
    "treino_e_teste_1" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "recebeu_outro_leite", "usou_bico_artificial", "nivel_inseguranca_alimentar", "usou_bomba_extracao", "inicio_prenatal", "reside_com_parceiro", "zona_residencial", "regiao_residencia", "recebe_auxilio_governamental", "busca_informacao_aleitamento", "tempo_ate_primeira_mamada", "faixa_etaria_mae"],
    "treino_e_teste_2" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "usou_bico_artificial", "recebeu_outro_leite", "zona_residencial", "usou_sondinha_relactacao", "faixa_etaria_mae", "usou_bomba_extracao", "regiao_residencia", 
                          "nivel_inseguranca_alimentar", "inicio_prenatal", "escolaridade_mae", "tempo_ate_primeira_mamada", "faixa_consultas_prenatal"],
    "treino_e_teste_3" : ["historico_uso_chupeta", "usou_mamadeira", "oferta_outros_liquidos", "recebeu_outro_leite", "faixa_etaria_mae", "regiao_residencia", "usou_bico_artificial", "nivel_inseguranca_alimentar", "faixa_renda_familiar", "escolaridade_mae", 
                          "usou_bomba_extracao", "busca_informacao_aleitamento", "tempo_ate_primeira_mamada", "raca_cor_mae", "inicio_prenatal"],
    "treino_e_teste_4" : ["historico_uso_chupeta", "faixa_renda_familiar", "busca_informacao_aleitamento", "classificacao_peso_nascimento", "zona_residencial", "regiao_residencia", "usou_sondinha_relactacao", "usou_copinho", "qtd_filhos_vivos", "tipo_parto", 
                          "usou_concha_amamentacao", "oferta_outros_liquidos", "usou_protetor_mamilo", "usou_bomba_extracao", "usou_bico_artificial"],
    "treino_e_teste_5" : ["historico_uso_chupeta", "regiao_residencia", "usou_mamadeira", "busca_informacao_aleitamento", "oferta_outros_liquidos", "recebeu_outro_leite", "escolaridade_mae", "nivel_inseguranca_alimentar", "inicio_prenatal", "recebe_auxilio_governamental", 
                          "raca_cor_mae", "usou_bico_artificial", "usou_bomba_extracao", "adequacao_peso_idade_gestacional", "faixa_renda_familiar"]
}

# 5. Iterar sobre as coleções para filtrar e salvar os CSVs
base_output_dir = 'selecao_dados/qtde_atributos_10/'

for nome_colecao, atributos in colecao_10_atributos.items():
    
    # Criar um subdiretório específico para a coleção atual
    # Ex: 'selecao_dados/qtde_atributos_10/treino_e_teste_1/'
    output_dir = os.path.join(base_output_dir, nome_colecao)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar os DataFrames usando apenas as colunas da lista atual
    X_treino_filtrado = X_treino[atributos]
    X_teste_filtrado = X_teste[atributos]
    
    # Juntar o X filtrado com o y correspondente
    # A concatenação usa o índice original das linhas, mantendo a integridade dos dados
    dataset_treino = pd.concat([X_treino_filtrado, y_treino], axis=1)
    dataset_teste = pd.concat([X_teste_filtrado, y_teste], axis=1)
    
    # Definir o caminho dos arquivos
    caminho_treino = os.path.join(output_dir, 'dataset_treino.csv')
    caminho_teste = os.path.join(output_dir, 'dataset_teste.csv')
    
    # Exportar para CSV sem a coluna de índice do Pandas
    dataset_treino.to_csv(caminho_treino, index=False)
    dataset_teste.to_csv(caminho_teste, index=False)
    
    print(f"Arquivos gerados com sucesso para a coleção: {nome_colecao}")
    
# 5. Iterar sobre as coleções para filtrar e salvar os CSVs
base_output_dir = 'selecao_dados/qtde_atributos_15/'

for nome_colecao, atributos in colecao_15_atributos.items():
    
    # Criar um subdiretório específico para a coleção atual
    # Ex: 'selecao_dados/qtde_atributos_10/treino_e_teste_1/'
    output_dir = os.path.join(base_output_dir, nome_colecao)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar os DataFrames usando apenas as colunas da lista atual
    X_treino_filtrado = X_treino[atributos]
    X_teste_filtrado = X_teste[atributos]
    
    # Juntar o X filtrado com o y correspondente
    # A concatenação usa o índice original das linhas, mantendo a integridade dos dados
    dataset_treino = pd.concat([X_treino_filtrado, y_treino], axis=1)
    dataset_teste = pd.concat([X_teste_filtrado, y_teste], axis=1)
    
    # Definir o caminho dos arquivos
    caminho_treino = os.path.join(output_dir, 'dataset_treino.csv')
    caminho_teste = os.path.join(output_dir, 'dataset_teste.csv')
    
    # Exportar para CSV sem a coluna de índice do Pandas
    dataset_treino.to_csv(caminho_treino, index=False)
    dataset_teste.to_csv(caminho_teste, index=False)
    
    print(f"Arquivos gerados com sucesso para a coleção: {nome_colecao}")

print("\nProcessamento concluído: Todos os subconjuntos de atributos foram separados e salvos!")