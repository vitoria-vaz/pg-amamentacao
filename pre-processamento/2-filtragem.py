import pandas as pd

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_eng.csv'
CAMINHO_SAIDA   = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_filtrado.csv'

ATRIBUTOS_PARA_EXCLUIR = [
    'tempo_amament_meses', 'b05a_idade_em_meses', 'id_anon', 'k28_rec', 'b02_sexo', 'idade_filho', 'amament_cat',
    'gestacoes_cat', 'k04_prenatal_semanas', 'h01_semanas_gravidez', 'k05_prenatal_consultas', 'inicio_precoce_amamentacao',
    'k248_utilizou_nao', 'k249_utilizou_nao_sabe'
]

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')
print(f"Dataset original carregado com sucesso! Total de atributos: {df.shape[1]}")

# ==========================================
# 3. PRÉ-PROCESSAMENTO (Filtragem de Domínio)
# ==========================================
# Garante que só tenta excluir colunas que realmente existem na base atual
colunas_excluir = [col for col in ATRIBUTOS_PARA_EXCLUIR if col in df.columns]
df_filtrado = df.drop(columns=colunas_excluir)
print(f"Sucesso: {len(colunas_excluir)} atributos foram removidos da base de dados.")

# ==========================================
# 4. SALVAMENTO DOS DADOS
# ==========================================
df_filtrado.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
print(f"Dataset filtrado (agora com {df_filtrado.shape[1]} atributos) salvo em:\n{CAMINHO_SAIDA}")