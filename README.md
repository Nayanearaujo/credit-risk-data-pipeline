<div align="center">

# Pipeline de Risco de Crédito

### Engenharia de Dados e Machine Learning para Concessão de Crédito

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![SQL](https://img.shields.io/badge/SQL-DuckDB-FFCA28?logo=duckdb&logoColor=black)](https://duckdb.org)
[![ML](https://img.shields.io/badge/ML-XGBoost-FF6600?logo=xgboost)](https://xgboost.readthedocs.io)
[![Dashboard](https://img.shields.io/badge/Dashboard-Online_no_Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://credit-risk-data-pipeline.streamlit.app/)
[![CI](https://github.com/Nayanearaujo/credit-risk-data-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Nayanearaujo/credit-risk-data-pipeline/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Autora:** [Nayane Araújo](https://github.com/Nayanearaujo) · Pós-Graduação em Engenharia de Dados

</div>

---

## Contexto de Negócio

Em concessão de crédito, equilibrar aprovação de pedidos com controle de inadimplência é o principal desafio operacional. Cada aprovação indevida gera perda direta de capital, enquanto uma política excessivamente conservadora recusa clientes lucrativos.

Este repositório implementa o ciclo de dados completo para apoiar essa decisão:
1. Ingestão e padronização dos dados brutos com rastreabilidade por camadas (Medallion)
2. Modelagem dimensional para consultas analíticas e agregações em SQL
3. Modelagem preditiva com validação cruzada, tuning de hiperparâmetros e calibração de probabilidades
4. Interface interativa com simulador de crédito em tempo real

*Nota de contexto sobre os dados:* Os registros utilizados pertencem a um dataset acadêmico de risco de crédito (hospedado na OpenML e Kaggle), não correspondendo a transações reais de uma instituição bancária específica. Serve como ambiente controlado para desenho de arquitetura de dados e avaliação de modelos.

---

## Arquitetura Medallion (Bronze -> Silver -> Gold)

O fluxo de dados segue a separação em três camadas para garantir reprodutibilidade e qualidade:

```
+-------------------------------------------------------------------+
|                       FONTES DE DADOS                             |
|   OpenML (ID 43454, CC0)            API Banco Central (Selic)     |
+--------------------+----------------------------------------------+
                     |
                     v
+------------------------------------+
|          BRONZE LAYER              |
|  - Ingestao fiel a origem (OpenML) |
|  - Fallback automatico via mirror  |
|  - Formato: CSV                    |
+--------------------+---------------+
                     |  Python (Pandas)
                     v
+------------------------------------+
|          SILVER LAYER              |
|  - Deduplicacao e tipagem          |
|  - Tratamento de nulos por grade   |
|  - Validacao de ranges de negocio  |
|  - Metadados de auditoria          |
|  - Formato: CSV + DuckDB           |
+--------------------+---------------+
                     |  SQL + Python
                     v
+------------------------------------+
|          GOLD LAYER                |
|  - Star Schema (fato e dimensoes)  |
|  - Agregacoes de negocio           |
|  - Features e scaler para modelo   |
|  - Formato: CSV + DuckDB           |
+--------+---------------------------+
         |
         +---> Modelos ML (XGBoost Tuned + Baseline)
         +---> Calibracao de Probabilidades (Brier Score)
         +---> Dashboard Interativo Streamlit
```

---

## Stack Tecnológica

| Categoria | Tecnologia | Finalidade no Projeto |
|---|---|---|
| Linguagem | Python 3.11 | Ingestão, transformações e modelagem preditiva |
| Banco / Analytics | DuckDB + PostgreSQL | Banco colunar in-process para analytics local e suporte a Postgres |
| Ingestão | OpenML API + Requests | Download oficial do dataset e coleta da Selic (API BCB) |
| Engenharia de Dados | Pandas + SQLAlchemy | Limpeza por regras de negócio e tipagem |
| Machine Learning | Scikit-Learn + XGBoost | Classificação binária com validação cruzada e tuning |
| Calibração | Scikit-Learn (CalibratedClassifierCV) | Reliability curve e otimização do Brier Score |
| Visualização | Streamlit + Plotly | Dashboard gerencial e simulador de propostas |
| Testes e Qualidade | Pytest, Flake8, Black | Testes unitários de produção e checagem estática no CI |
| Integração Contínua | GitHub Actions | Workflows automatizados de teste e linting |

---

## Resultados e Métricas

### Performance dos Modelos

A avaliação dos modelos foi realizada com validação cruzada estratificada em 5 folds no conjunto de treino balanceado (SMOTE), seguida de avaliação no conjunto de teste independente (hold-out) com as probabilidades avaliadas pelo Brier Score:

| Modelo | AUC-ROC (CV 5-Fold) | AUC-ROC (Teste) | F1-Score | Precisão | Recall | KS Statistic | Brier Score |
|---|---|---|---|---|---|---|---|
| **XGBoost (Tuned)** | **0.9840 ± 0.0013** | **0.9388** | **0.8220** | **0.9205** | **0.7426** | **0.7390** | **0.0585** |
| Random Forest | 0.9643 ± 0.0016 | 0.9092 | 0.7771 | 0.8150 | 0.7426 | 0.7037 | 0.0822 |
| Regressão Logística | 0.9069 ± 0.0018 | 0.8295 | 0.6038 | 0.5476 | 0.6728 | 0.5281 | 0.1399 |

Os hiperparâmetros otimizados para o XGBoost vencedor via busca aleatória (`max_depth: 6`, `learning_rate: 0.08`, `n_estimators: 200`, `subsample: 0.95`, `colsample_bytree: 0.85`) encontram-se registrados em `data/gold/best_params.json` (ADR-006).

### Calibração das Probabilidades

Em análise de risco, o ordenamento do ranking precisa ser acompanhado por probabilidades bem calibradas: uma probabilidade estimada de 20% deve significar que, historicamente, 20 em cada 100 contratos similares inadimpliram. O XGBoost alcançou Brier Score de 0.0585 na base de teste, e o gráfico de calibração correspondente está salvo em `docs/calibration_curve.png`.

### Observações dos Dados

O que mais chamou minha atenção ao analisar os dados foi como o percentual de comprometimento de renda (`loan_percent_income`) tem um efeito multiplicador quando associado a apontamentos anteriores (`cb_person_default_on_file`). Quando o solicitante já possui registro prévio e pede um empréstimo que consome mais de 30% da sua renda anual, a taxa observada salta para mais de 65%. Em contrapartida, tomadores com boa nota de crédito (Grade A ou B) mantêm a taxa média de inadimplência abaixo de 12%, mesmo com prazos mais longos.

Outro achado relevante na análise por segmentação foi na faixa etária entre 18 e 25 anos, onde a taxa de default observada atingiu 26%, sugerindo que critérios de histórico financeiro curto demandam limites iniciais mais cautelosos.

---

## Como Executar

### Pré-requisitos
- Python 3.11 instalado
- Gerenciador de pacotes pip

### 1. Clonar o repositório
```bash
git clone https://github.com/Nayanearaujo/credit-risk-data-pipeline.git
cd credit-risk-data-pipeline
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
cp -n .env.example .env || true
```

### 3. Rodar o pipeline completo
```bash
python run_pipeline.py
```
O script executa automaticamente a ingestão da OpenML (caso não haja arquivo Bronze local), as regras de limpeza da camada Silver, a modelagem dimensional Gold, a engenharia de features, o treinamento dos modelos com validação cruzada e o teste dos artefatos gerados.

### 4. Iniciar o dashboard
```bash
streamlit run src/dashboard/app.py
```
Acesse em `http://localhost:8501` para navegar pela Visão Geral, Análise de Segmentos e Simulador de Crédito.

### 5. Executar os testes
```bash
pytest tests/ -v
```

### 6. Deploy e Sincronização em Produção
A aplicação está implantada no [Streamlit Community Cloud](https://credit-risk-data-pipeline.streamlit.app/).
- **Artefatos Versionados:** Os artefatos pré-computados (`data/gold/fact_loans.csv`, `data/gold/model_comparison.csv`, `src/models/best_model.pkl`, `src/models/scaler.pkl` e `src/models/feature_names.csv`) são rastreados no Git para inicialização instantânea e determinística no ambiente serverless.
- **Ciclo de Atualização:** Ao rodar `python run_pipeline.py`, os modelos e agregações são retreinados e recalculados. Um `git push origin main` dispara o redeploy automático no Streamlit Cloud via webhook nativo do GitHub.
- **Disponibilidade Contínua:** Um workflow agendado (`.github/workflows/keep_alive.yml`) mantém o aplicativo ativo contra sleep automático da plataforma.

---

## Estrutura do Projeto

```
credit-risk-data-pipeline/
|
|-- README.md
|-- requirements.txt               <- Dependencias do projeto
|-- requirements-ci.txt            <- Dependencias para o GitHub Actions
|-- Makefile                       <- Comandos rapidos de execucao
|-- run_pipeline.py                <- Orquestrador do pipeline de ponta a ponta
|
|-- data/
|   |-- bronze/                    <- Dados brutos (credit_risk_raw.csv)
|   |-- silver/                    <- Base limpa e validada (credit_risk_clean.csv)
|   +-- gold/                      <- Star schema, agregacoes e metricas
|
|-- src/
|   |-- ingestion/
|   |   |-- ingest_openml.py       <- Ingestao oficial via OpenML (ID 43454) com fallback
|   |   |-- ingest_kaggle.py       <- Wrapper de compatibilidade
|   |   +-- ingest_bcb_api.py      <- Coleta da Taxa Selic no Banco Central
|   |-- transformation/
|   |   |-- silver_transform.py    <- Deduplicacao, tipagem, nulos e limites de negocio
|   |   +-- gold_transform.py      <- Modelagem dimensional (fact_loans, dimensoes)
|   |-- models/
|   |   |-- feature_engineering.py <- Features derivadas, encoding e split balanceado
|   |   |-- train_model.py         <- Treinamento com 5-fold CV e tuning XGBoost
|   |   |-- calibration.py         <- Avaliacao de calibracao e Brier Score
|   |   +-- evaluate_model.py      <- Curva ROC, matriz de confusao e metricas
|   |-- dashboard/
|   |   +-- app.py                 <- Aplicacao Streamlit e simulador alinhado
|   +-- utils/
|       |-- db.py                  <- Conexoes DuckDB e PostgreSQL
|       +-- logger.py              <- Configuracao padronizada do Loguru
|
|-- scripts/
|   +-- dev/
|       +-- gera_dados_sinteticos.py <- Gerador de dados sinteticos para dev offline
|
|-- tests/
|   |-- test_ingestion.py          <- Testes unitarios da camada Bronze
|   |-- test_transformation.py     <- Testes unitarios das regras Silver
|   |-- test_simulator_inference.py <- Validacao de inferencia e score do simulador
|   +-- test_integration.py        <- Teste de integracao end-to-end com fixture
|
|-- .github/workflows/
|   |-- ci.yml                     <- Pipeline de CI (Black, Flake8, pytest)
|   +-- keep_alive.yml             <- Rotina de disponibilidade para o Streamlit Cloud
|
+-- docs/
    |-- decisions.md               <- Registro de decisoes tecnicas (ADRs 001 a 006)
    |-- calibration_curve.png      <- Curva de confiabilidade do modelo campeao
    |-- confusion_matrix.png
    +-- roc_curve.png
```

---

## Dataset e Fontes

1. **OpenML Credit Risk Dataset (ID: 43454)**
   - Fonte oficial citável: [openml.org/search?type=data&id=43454](https://www.openml.org/search?type=data&id=43454)
   - Licença: CC0 (Domínio Público)
   - 32.581 registros de propostas de empréstimo e 12 variáveis
   - Taxa natural de inadimplência observada: ~21.8%
   - Cópia espelho no GitHub mantida como mecanismo automático de fallback

2. **API do Banco Central do Brasil (SGS)**
   - Série 432: Taxa Selic acumulada e diária
   - Contexto macroeconômico integrado via API REST pública

---

## Próximos Passos

- [x] Ingestão oficial via OpenML e fallback automático
- [x] Validação cruzada estratificada em 5 folds e tuning de hiperparâmetros
- [x] Calibração de probabilidades e cálculo do Brier Score
- [x] Teste de integração ponta a ponta com fixture de dados
- [x] Deploy público contínuo no Streamlit Community Cloud
- [ ] Orquestração programada com Apache Airflow
- [ ] Monitoramento contínuo de drift com Evidently AI

---

## Autora

**Nayane Araújo**
Pós-Graduanda em Engenharia de Dados

[![GitHub](https://img.shields.io/badge/GitHub-Nayanearaujo-181717?logo=github&logoColor=white)](https://github.com/Nayanearaujo)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-0077B5?logo=linkedin&logoColor=white)](https://linkedin.com/in/nayanearaujo)
