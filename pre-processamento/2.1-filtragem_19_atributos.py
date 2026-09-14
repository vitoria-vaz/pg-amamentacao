import pandas as pd

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\selecao_limpeza\dataset_amamentacao_eng_2.csv'
CAMINHO_SAIDA   = 'selecao_limpeza/dataset_filtrado_20_atributos.csv'

ATRIBUTOS_PARA_MANTER = [ 
    'exposicao_mamadeira', 'b05a_idade_em_meses', 'h05_chupeta_usou', 'busca_info_aleitamento', 
    'k20_doou', 'h02_peso', 'q01_recebe_beneficio', 'h13_diarreia', 'p05_comodos', 
    'k06_peso_engravidar', 'vd_zwaz', 'utilizou_apoio_amamentacao', 'vd_imc_mae', 
    'bb04_idade_da_mae', 'h01_semanas_gravidez', 'vd_ien_escore', 'k04_prenatal_semanas', 
    'k08_quilos', 'tempo_primeira_mamada_horas', 'aleitamento_materno_exclusivo'
]

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')
print(f"Dataset original carregado com sucesso! Total de atributos originais: {df.shape[1]}")

# ---------------------------------------------------------
# Remover mães que adotaram (d06 == 1)
# ---------------------------------------------------------
df = df[df['d06_relacao_responsavel'] != 1.0]
print(f"Instâncias com mães que adotaram removidas. Linhas restantes: {len(df)}")

# ==========================================
# 3. PRÉ-PROCESSAMENTO (Filtragem de Domínio)
# ==========================================
# Garante que só tentará manter as colunas que realmente existem no dataframe atual
colunas_presentes = [col for col in ATRIBUTOS_PARA_MANTER if col in df.columns]

# Verificação de segurança: avisa se digitou algum nome errado ou se faltou alguma coluna
colunas_ausentes = set(ATRIBUTOS_PARA_MANTER) - set(colunas_presentes)
if colunas_ausentes:
    print(f"⚠️ Atenção: As seguintes colunas não foram encontradas no CSV e serão ignoradas:\n{colunas_ausentes}")

# Seleciona apenas as colunas desejadas
df_filtrado = df[colunas_presentes]
print(f"\nSucesso: O dataset agora possui apenas {df_filtrado.shape[1]} atributos.")

# ==========================================
# 4. SALVAMENTO DOS DADOS
# ==========================================
df_filtrado.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
print(f"Dataset filtrado salvo em:\n{CAMINHO_SAIDA}")