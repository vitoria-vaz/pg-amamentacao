# Passo 1: Importar a biblioteca necessária
# O pandas é utilizado para manipulação e análise de dados tabulares.
import pandas as pd

def explorar_categorias(caminho_do_ficheiro, limite_categorias=30):
    """
    Função para ler um dataset e imprimir as categorias de cada atributo.
    
    Parâmetros:
    - caminho_do_ficheiro: O nome ou caminho do teu ficheiro (ex: 'dados.csv').
    - limite_categorias: O número máximo de categorias únicas para imprimirmos. 
                         Isto evita imprimir colunas com milhares de números ou IDs.
    """
    print(f"A carregar o dataset: {caminho_do_ficheiro}...\n")
    
    try:
        # Passo 2: Carregar o dataset
        # O pd.read_csv lê o ficheiro e transforma-o num 'DataFrame' (uma tabela de dados do Pandas)
        df = pd.read_csv(caminho_do_ficheiro)
        
        print("Dataset carregado com sucesso!")
        print("-" * 40)
        
        # Passo 3: Percorrer cada coluna do dataset
        for coluna in df.columns:
            
            # Passo 4: Obter os valores únicos da coluna atual
            # O método .dropna() ignora valores nulos/vazios
            # O método .unique() devolve apenas os valores que não se repetem
            categorias = df[coluna].dropna().unique()
            
            # Conta quantas categorias existem
            quantidade = len(categorias)
            
            # Passo 5: Decidir se imprimimos as categorias com base no limite
            if quantidade <= limite_categorias:
                print(f"Atributo: '{coluna}' (Possui {quantidade} categorias)")
                print(f"Categorias: {list(categorias)}")
                print("-" * 40)
            else:
                # Se tiver muitas categorias, mostramos apenas a quantidade para evitar poluição visual
                print(f"Atributo: '{coluna}' (Possui {quantidade} valores únicos - ignorado por exceder o limite de {limite_categorias})")
                print("-" * 40)
                
    except FileNotFoundError:
        # Tratamento de erro caso o ficheiro não seja encontrado
        print(f"ERRO: O ficheiro '{caminho_do_ficheiro}' não foi encontrado. Verifica o nome e o caminho.")

# ==========================================
# ÁREA DE EXECUÇÃO
# ==========================================
# Variáveis que podes (e deves) ajustar:
NOME_DO_FICHEIRO = 'dataset/dataset_pos_processamento_2.csv' # <-- Substitui pelo nome real do teu ficheiro
LIMITE = 20 # <-- Altera este valor se quiseres ver colunas com mais ou menos categorias

# Chamada da função para iniciar o programa
explorar_categorias(NOME_DO_FICHEIRO, limite_categorias=LIMITE)