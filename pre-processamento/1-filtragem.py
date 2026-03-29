import pandas as pd

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS (Constantes)
# ==========================================
CAMINHO_ENTRADA = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao.csv'
CAMINHO_SAIDA   = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_filtrado.csv'

ATRIBUTOS_PARA_EXCLUIR = [
    'k18_somente', 'k19_somente_medida', 'tempo_amament_meses', 
    'b05a_idade_em_meses', 'd06_relacao_responsavel', 'id_anon',
    'k04_prenatal_semanas', 'h01_semanas_gravidez', 'k05_prenatal_consultas',
    'k12_tempo', 'k13_tempo_medida', 'k28_aleitamento'
]

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
df = pd.read_csv(CAMINHO_ENTRADA, encoding='utf-8')
print(f"Dataset original carregado com sucesso! Total de atributos: {df.shape[1]}")

# ==========================================
# 3. PRÉ-PROCESSAMENTO (Filtragem de Domínio)
# ==========================================
df_filtrado = df.drop(columns=ATRIBUTOS_PARA_EXCLUIR)
print(f"Sucesso: {len(ATRIBUTOS_PARA_EXCLUIR)} atributos foram removidos da base de dados.")

# ==========================================
# 4. SALVAMENTO DOS DADOS
# ==========================================
df_filtrado.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8')
print(f"Dataset filtrado (agora com {df_filtrado.shape[1]} atributos) salvo em:\n{CAMINHO_SAIDA}")