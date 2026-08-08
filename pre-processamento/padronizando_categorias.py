import pandas as pd
import unicodedata
import re

caminho = r'C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset_amamentacao_renomeado.csv'
arquivo_saida = r"C:\Users\vitoria-vaz\estudos\UFU\projeto-graduacao\pg-amamentacao\dataset\dataset_amamentacao_categorias_padronizadas.csv"

def padronizar_categoria(texto):
    """
    Função para padronizar o texto das categorias preservando numerais.
    """
    if pd.isna(texto):
        return texto
    
    texto = str(texto).strip()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = texto.replace(' ', '_')
    
    # Atualização: Mantendo vírgulas e pontos para não distorcer valores numéricos
    texto = re.sub(r'[^\w\-\<\>\=\+\,\.]', '', texto)
    
    texto_minusculo = texto.lower()
    
    return texto_minusculo

# Carregar o dataset
df = pd.read_csv(caminho)

# 1. Mapeamento Explicito para a Faixa de Renda
# Substitua as chaves pelos valores brutos exatos que aparecem no seu CSV
mapeamento_renda = {
    'Até R$ 1.000,00' : '<=1000',
    'De R$ 1.001,00 até R$ 2.000,00' : '>1000_a_<=2000',
    'De R$ 2.001,00 até R$ 3.000,00' : '>2000_a_<=3000',
    'De R$ 3.001,00 até R$ 5.000,00' : '>3000_a_<=5000',
    'R$ 5.001,00 ou mais' : '>5000',
    'Sem renda' : 'Sem_renda',
}

if 'faixa_renda_familiar' in df.columns:
    df['faixa_renda_familiar'] = df['faixa_renda_familiar'].replace(mapeamento_renda)

# 2. Aplicar a limpeza automática nas demais colunas
colunas_alvo = [
    'regiao_residencia', 'zona_residencial', 'alvo_sucesso_ame_6m',
    'escolaridade_mae', 'qtd_filhos_vivos', 'tipo_parto',
    'historico_uso_chupeta', 'faixa_etaria_mae', 'inicio_prenatal',
    'reside_com_parceiro', 'situacao_laboral_mae', 'realizou_prenatal',
    'recebeu_outro_leite', 'oferta_outros_liquidos', 'usou_concha_amamentacao',
    'usou_protetor_mamilo', 'usou_bico_artificial', 'usou_bomba_extracao',
    'usou_mamadeira', 'usou_sondinha_relactacao', 'usou_copinho',
    'busca_informacao_aleitamento', 'faixa_consultas_prenatal',
    'classificacao_peso_nascimento', 'adequacao_peso_idade_gestacional',
    'recebe_auxilio_governamental', 'tempo_ate_primeira_mamada', 
    'nivel_inseguranca_alimentar', 'raca_cor_mae'
]

for col in colunas_alvo:
    if col in df.columns:
        df[col] = df[col].apply(padronizar_categoria)
        
coluna_desfecho = 'alvo_sucesso_ame_6m'
cols = [c for c in df.columns if c != coluna_desfecho] + [coluna_desfecho]
df = df[cols]

# Salvar o dataset corrigido
df.to_csv(arquivo_saida, index=False)

print(f"\nTransformação concluída! Dataset salvo como: {arquivo_saida}")