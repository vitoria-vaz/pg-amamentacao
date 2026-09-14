import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\criancas_maiores_6_meses_classificadas.csv'
CAMINHO_SAIDA   = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\dataset_amamentacao_eng_2.csv'

# ==========================================
# 2. FUNÇÕES DE ENGENHARIA DE ATRIBUTOS
# ==========================================

def consolidar_apoio_amamentacao(df):
    """Cria a variável utilizou_apoio_amamentacao e exclui as colunas originais."""
    print("\n--- 1. Consolidando Apoio à Amamentação ---")
    
    apoio_vars = ['k241_utilizou_concha', 'k242_utilizou_protetor', 'k243_utilizou_bico', 
                  'k244_utilizou_bomba', 'k245_utilizou_mamadeira']
    
    vars_to_exclude = ['k24_utilizou', 'k246_utilizou_sondinha', 'k247_utilizou_copo', 
                       'k248_utilizou_nao', 'k249_utilizou_nao_sabe'] + apoio_vars

    existing_apoio = [var for var in apoio_vars if var in df.columns]
    
    if existing_apoio:
        # Cria máscara booleana se alguma das variáveis for 'Sim', e converte para int (0/1)
        utilizou_apoio = (df[existing_apoio] == 'Sim').any(axis=1).astype(int)
        
        # Insere na posição da primeira variável encontrada
        insert_pos = df.columns.get_loc(existing_apoio[0])
        df.insert(insert_pos, 'utilizou_apoio_amamentacao', utilizou_apoio)
        
        # Estatísticas
        pct_apoio = utilizou_apoio.mean() * 100
        print(f"Nova variável gerada: utilizou_apoio_amamentacao ({pct_apoio:.1f}% utilizaram algum apoio)")

    # Exclusão das variáveis originais
    colunas_para_remover = [var for var in vars_to_exclude if var in df.columns]
    df.drop(columns=colunas_para_remover, inplace=True)
    print(f"Total de variáveis originais excluídas: {len(colunas_para_remover)}")
    
    return df


def classificar_busca_informacao(df):
    """Cria variáveis binárias para busca de informações sobre aleitamento e alimentação."""
    print("\n--- 2. Classificando Busca de Informações ---")
    respostas_positivas = ['Muito', 'Pouco', 'Mais ou menos']
    
    # Aleitamento
    if 'k28_aleitamento' in df.columns:
        busca_aleit = df['k28_aleitamento'].isin(respostas_positivas).astype(int)
        df.insert(df.columns.get_loc('k28_aleitamento'), 'busca_info_aleitamento', busca_aleit)
        df.drop(columns=['k28_aleitamento'], inplace=True)
        print(f"Nova variável: busca_info_aleitamento ({(busca_aleit.mean() * 100):.1f}% buscaram)")

    # Alimentação
    if 'k29_alimentacao' in df.columns:
        busca_alim = df['k29_alimentacao'].isin(respostas_positivas).astype(int)
        df.insert(df.columns.get_loc('k29_alimentacao'), 'busca_info_alimentacao', busca_alim)
        df.drop(columns=['k29_alimentacao'], inplace=True)
        print(f"Nova variável: busca_info_alimentacao ({(busca_alim.mean() * 100):.1f}% buscaram)")
        
    return df


def padronizar_tempo_mamada(df):
    """Converte o tempo da primeira mamada para horas."""
    print("\n--- 3. Padronizando Tempo da Primeira Mamada ---")
    if 'k12_tempo' in df.columns and 'k13_tempo_medida' in df.columns:
        tempo_horas = df['k12_tempo'].copy()
        
        # Multiplica por 24 onde a medida for 'Dias'
        mask_dias = df['k13_tempo_medida'] == 'Dias'
        tempo_horas.loc[mask_dias] = tempo_horas.loc[mask_dias] * 24
        
        df.insert(df.columns.get_loc('k12_tempo'), 'tempo_primeira_mamada_horas', tempo_horas)
        df.drop(columns=['k12_tempo', 'k13_tempo_medida'], inplace=True)
        
        print(f"Nova variável: tempo_primeira_mamada_horas")
        print(f"Valores Ausentes (Missing): {tempo_horas.isna().sum()}")
        
    return df


def classificar_exposicao_mamadeira(df):
    """Cria variável binária para exposição à mamadeira."""
    print("\n--- 4. Classificando Exposição à Mamadeira ---")
    if 'k25_mamadeira' in df.columns:
        exposicao = df['k25_mamadeira'].isin(['Sim, ainda usa', 'Sim, já usou mas não usa mais']).astype(int)
        df.insert(df.columns.get_loc('k25_mamadeira'), 'exposicao_mamadeira', exposicao)
        df.drop(columns=['k25_mamadeira'], inplace=True)
        
        pct_exposicao = exposicao.mean() * 100
        print(f"Nova variável: exposicao_mamadeira ({pct_exposicao:.1f}% expostas)")
        
    return df

# ==========================================
# 3. PIPELINE DE EXECUÇÃO PRINCIPAL
# ==========================================

def executar_pipeline():
    print("="*50)
    print("INICIANDO PIPELINE DE ENGENHARIA DE ATRIBUTOS")
    print("="*50)
    
    # 1. Carregamento
    try:
        df = pd.read_csv(CAMINHO_ENTRADA)
        print(f"Dataset carregado com sucesso. Formato inicial: {df.shape}")
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {CAMINHO_ENTRADA}")
        return

    # 2. Aplicação das Transformações
    df = consolidar_apoio_amamentacao(df)
    df = classificar_busca_informacao(df)
    df = padronizar_tempo_mamada(df)
    df = classificar_exposicao_mamadeira(df)

    # 3. Salvamento
    print("\n" + "="*50)
    df.to_csv(CAMINHO_SAIDA, index=False)
    print(f"🚀 Pipeline concluído com sucesso!")
    print(f"Dataset final salvo em: {CAMINHO_SAIDA}")
    print(f"Formato final: {df.shape}")
    print("="*50)

# Bloco de execução padrão do Python
if __name__ == "__main__":
    executar_pipeline()