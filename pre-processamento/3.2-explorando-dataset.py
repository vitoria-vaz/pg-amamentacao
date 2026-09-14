import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\criancas_maiores_6meses.csv'
CAMINHO_ENTRADA2 = 'selecao_limpeza/dataset_amamentacao.csv'

# Caminho para salvar o arquivo de texto com os resultados
CAMINHO_SAIDA_TXT = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\analise_valores_unicos.txt'

# Nomes dos arquivos de imagem gerados
CAMINHO_GRAFICO_NULOS_TODOS = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\graficos_distribuicao\tentativa_2\contagem_valores_nulos_todos.jpg'
CAMINHO_GRAFICO_TIPOS = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\graficos_distribuicao\tentativa_2\contagem_tipos_dados.jpg'

# ==========================================
# 2. CARREGAMENTO DOS DADOS E INFORMAÇÕES
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')
df2 = pd.read_csv(CAMINHO_ENTRADA2, encoding='utf-8')

# ==========================================
# 3. ANÁLISE EXPLORATÓRIA: VALORES 
# ==========================================
# Abre o arquivo TXT em modo de escrita ('w'). O encoding 'utf-8' evita problemas com acentos.
with open(CAMINHO_SAIDA_TXT, 'w', encoding='utf-8') as arquivo_txt:
    
    # O parâmetro file=arquivo_txt redireciona o texto do console para o arquivo
    print("Análise de Valores Únicos por Atributo:\n" + "="*40, file=arquivo_txt)

    # Itera sobre cada coluna do dataset
    for coluna in df.columns:
        valores_unicos = df[coluna].unique()
        quantidade = df[coluna].nunique(dropna=False) # dropna=False inclui valores nulos (NaN) na contagem
        
        print(f"\nAtributo: '{coluna}'", file=arquivo_txt)
        print(f"Total de valores únicos: {quantidade}", file=arquivo_txt)
        
        # Condição para não imprimir todos os valores caso a coluna seja muito cardinal
        if quantidade <= 20:
            print(f"Valores: {valores_unicos}", file=arquivo_txt)
        else:
            print(f"Valores (amostra de 5): {valores_unicos[:5]} ... [Muitos valores para exibir]", file=arquivo_txt)

# Um print normal no final apenas para avisar no terminal que o processo terminou
print(f"Análise concluída com sucesso! Os resultados foram salvos em:\n{CAMINHO_SAIDA_TXT}")