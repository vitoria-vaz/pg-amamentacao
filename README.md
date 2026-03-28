# 🍼 Previsão de Interrupção Precoce da Amamentação

**Projeto de Graduação em Bacharelado em Sistemas de Informação - Universidade Federal de Uberlândia**

## 📌 Sobre o Projeto
A amamentação é fundamental para a saúde e o desenvolvimento infantil. No entanto, diversos fatores demográficos, clínicos e sociais podem levar à sua interrupção precoce. 

Este projeto aplica técnicas de **Ciência de Dados e Aprendizado de Máquina** para criar um modelo probabilístico capaz de prever o risco de desmame precoce. O objetivo é fornecer uma ferramenta de apoio à decisão que possa auxiliar profissionais de saúde na identificação antecipada de mães e bebês que necessitam de maior suporte.

## 🧠 Metodologia e Modelagem
O modelo principal escolhido para esta tarefa é a **Rede Bayesiana de Crença (Bayesian Belief Network - BBN)**. A escolha deste algoritmo se justifica por sua excelente capacidade de lidar com incertezas, trabalhar com probabilidades condicionais e oferecer alta explicabilidade (interpretabilidade) clínica.

### 🛠️ Etapas do Pipeline de Dados
1. **Limpeza e Filtragem:** Remoção de atributos não relevantes conforme indicação de especialistas do domínio (médicos/enfermeiros).
2. **Tratamento de Dados Ausentes:** Identificação e imputação de valores nulos.
3. **Discretização:** Transformação de variáveis contínuas em variáveis categóricas (faixas/bins), otimizando o processamento das Tabelas de Probabilidade Condicional (CPTs) da Rede Bayesiana.
4. **Seleção de Atributos:** Análise de associação (ex: Informação Mútua / V de Cramér) para selecionar as variáveis com maior poder preditivo.
5. **Treinamento e Avaliação:** Construção do grafo acíclico direcionado (DAG) e avaliação do modelo utilizando métricas de classificação (Acurácia, Precisão, Recall e Curva ROC).

## 🗂️ Conjunto de Dados (Dataset)
Os dados utilizados neste projeto contêm informações clínicas e demográficas. 
> **⚠️ Nota de Ética e Privacidade:** Em conformidade com a LGPD e as diretrizes de ética em pesquisa médica, todos os dados presentes neste repositório foram rigorosamente **anonimizados**. Nenhuma informação que permita a identificação pessoal dos pacientes está incluída.

## 💻 Tecnologias Utilizadas
* **Linguagem:** Python 3.x
* **Manipulação e Análise de Dados:** `pandas`, `numpy`
* **Pré-processamento e Métricas:** `scikit-learn`
* **Modelagem Probabilística:** `pgmpy` *(Probabilistic Graphical Models using Python)*
* **Visualização:** `matplotlib`, `seaborn`

## 🚀 Como Executar o Projeto

1. Clone este repositório:
   ```bash
   git clone https://github.com/vitoria-vaz/pg-amamentacao.git
