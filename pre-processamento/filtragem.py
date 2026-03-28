import pandas as pd

df = pd.read_csv('pg-amamentacao\dataset\dataset_amamentacao.csv', encoding='utf-8')

# ---------------------------------------------------------
# PASSO 1.1: Listando os atributos a serem excluídos
# ---------------------------------------------------------
atributos_para_excluir = ['k18_somente',
                          'k19_somente_medida',
                          'tempo_amament_meses',
                          'b05a_idade_em_meses',
                          'd06_relacao_responsavel',
                          'id_anon',
                          'k04_prenatal_semanas',
                          'h01_semanas_gravidez',
                          'k05_prenatal_consultas',
                          'k12_tempo',
                          'k13_tempo_medida',
                          'k28_aleitamento']

# ---------------------------------------------------------
# PASSO 1.2: Excluindo os atributos listados
# ---------------------------------------------------------
df_filtrado = df.drop(columns=atributos_para_excluir)
print("Atributos excluídos com sucesso!")
df_filtrado.to_csv('pg-amamentacao\dataset\dataset_amamentacao_filtrado.csv', index=False, encoding='utf-8')
print("Dataset filtrado salvo com sucesso!")

