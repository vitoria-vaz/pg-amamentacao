import pandas as pd

# Substitua pelo caminho onde seu dataset está salvo
caminho = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_pronto.csv'
df = pd.read_csv(caminho)

# Dicionário de mapeamento (De -> Para)
dicionario_renomeacao = {
    'a00_regiao': 'regiao_residencia',
    'a11_situacao': 'zona_residencial',
    'alvo_amamentacao': 'alvo_sucesso_ame_6m',
    'escolaridade_cat': 'escolaridade_mae',
    'filhos_vivos_cat': 'qtd_filhos_vivos',
    'h04_parto': 'tipo_parto',
    'h05_chupeta_usou': 'historico_uso_chupeta',
    'idade_mae_cat': 'faixa_etaria_mae',
    'inic_prenat': 'inicio_prenatal',
    'j04_vive': 'reside_com_parceiro',
    'j06_ocupacao': 'situacao_laboral_mae',
    'k03_prenatal': 'realizou_prenatal',
    'k15_recebeu': 'recebeu_outro_leite',
    'k16_liquido': 'oferta_outros_liquidos',
    'k241_utilizou_concha': 'usou_concha_amamentacao',
    'k242_utilizou_protetor': 'usou_protetor_mamilo',
    'k243_utilizou_bico': 'usou_bico_artificial',
    'k244_utilizou_bomba': 'usou_bomba_extracao',
    'k245_utilizou_mamadeira': 'usou_mamadeira',
    'k246_utilizou_sondinha': 'usou_sondinha_relactacao',
    'k247_utilizou_copo': 'usou_copinho',
    'k28_aleitamento': 'busca_informacao_aleitamento',
    'num_consultas': 'faixa_consultas_prenatal',
    'peso_cat': 'classificacao_peso_nascimento',
    'peso_ig': 'adequacao_peso_idade_gestacional',
    'q01_recebe_beneficio': 'recebe_auxilio_governamental',
    'q07_renda_faixa': 'faixa_renda_familiar',
    'tempo_1a_cat': 'tempo_ate_primeira_mamada',
    'vd_ebia_categ': 'nivel_inseguranca_alimentar'
}

# Renomeando as colunas
df.rename(columns=dicionario_renomeacao, inplace=True)

# Exportando a versão final
df.to_csv('dataset_amamentacao_renomeado.csv', index=False, encoding='utf-8')

print("Colunas renomeadas com sucesso!")