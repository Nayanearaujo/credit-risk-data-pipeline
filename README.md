<div align="center">

# 💳 Pipeline de Risco de Crédito

### Plataforma End-to-End de Engenharia e Ciência de Dados

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![SQL](https://img.shields.io/badge/SQL-DuckDB-FFCA28?logo=duckdb&logoColor=black)](https://duckdb.org)
[![ML](https://img.shields.io/badge/ML-XGBoost-FF6600?logo=xgboost)](https://xgboost.readthedocs.io)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![CI](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Autora:** [Nayane Araújo](https://github.com/Nayanearaujo) · Pós-Graduação em Engenharia de Dados

</div>

---

## 🎯 Problema de Negócio

> Em 2023, o Brasil registrou taxa de inadimplência de **6,2%** nas concessões de crédito pessoal (BCB/2023).
> Cada decisão de crédito errada representa prejuízo direto e perda de confiança do cliente.

Este projeto **simula a plataforma de dados de uma fintech/banco**, entregando:

1. **Pipeline ETL automatizado** que ingere, limpa e modela dados de crédito
2. **Modelo de Machine Learning** para prever probabilidade de inadimplência
3. **Dashboard interativo** para análise de risco por segmento

---

## 🏗️ Arquitetura — Medallion (Bronze → Silver → Gold)

```
+-------------------------------------------------------------------+
|                       FONTES DE DADOS                             |
|   Kaggle Credit Risk Dataset       API Banco Central (Selic)      |
+--------------------+----------------------------------------------+
                     |
                     v
+------------------------------------+
|          BRONZE LAYER              |
|  - Dados brutos sem alteracoes     |
|  - Preserva audit trail            |
|  - Formato: CSV                    |
+--------------------+---------------+
                     |  Python (Pandas)
                     v
+------------------------------------+
|          SILVER LAYER              |
|  - Limpeza e validacao             |
|  - Tipagem correta                 |
|  - Nulos tratados (mediana)        |
|  - Ranges validados                |
|  - Formato: CSV + DuckDB           |
+--------------------+---------------+
                     |  SQL + Python
                     v
+------------------------------------+
|          GOLD LAYER                |
|  - Star Schema (fatos/dimensoes)   |
|  - Agregacoes de negocio           |
|  - Features para ML                |
|  - Formato: CSV + DuckDB           |
+--------+---------------------------+
         |
         +---> Modelo de ML (XGBoost)
         +---> Dashboard Streamlit
         +---> Relatorios SQL
```

---

## 🛠️ Stack Tecnológica

| Categoria | Tecnologia | Finalidade |
|---|---|---|
| 🐍 Linguagem | Python 3.11 | ETL, ML, automação |
| 🗄️ SQL / Banco | DuckDB + PostgreSQL | Analytics local + produção |
| 🔄 ETL | Pandas + SQLAlchemy | Transformações por camada |
| 🤖 Machine Learning | Scikit-Learn + XGBoost | Classificação de risco |
| 📊 Stats | StatsModels | Análise estatística |
| 📈 Dashboard | Streamlit + Plotly | Visualização interativa |
| ⚖️ Balanceamento | Imbalanced-Learn (SMOTE) | Tratamento de classe desbalanceada |
| 🔧 CI/CD | GitHub Actions | Testes automatizados |
| 🧪 Testes | pytest + pytest-cov | Qualidade do código |
| ☁️ Cloud-ready | Estrutura S3/ADLS compatível | Azure/AWS/GCP |

---

## 📊 Resultados Principais

### 🏆 Performance do Modelo (XGBoost — Melhor Modelo)

| Métrica | Valor |
|---|---|
| **AUC-ROC** | ~0.93 |
| **F1-Score** | ~0.85 |
| **Precision** | ~0.82 |
| **Recall** | ~0.88 |
| **KS Statistic** | ~0.72 |

### 💡 Insights de Negócio

1. **Grade G tem 33% de inadimplência** — 3x mais que a Grade A (11%)
2. **Empréstimos para "Debt Consolidation" têm risco 40% maior** que os de educação
3. **Tomadores com histórico de inadimplência prévia** têm probabilidade 2.3x maior de inadimplir
4. **Comprometimento de renda > 30%** é o principal preditor de risco (SHAP values)
5. **Faixa etária 18-25** apresenta a maior taxa de inadimplência (26%)

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.11+
- pip

### 1. Clone o repositório
```bash
git clone https://github.com/Nayanearaujo/credit-risk-data-pipeline.git
cd credit-risk-data-pipeline
```

### 2. Configure o ambiente
```bash
make setup
# ou manualmente:
pip install -r requirements.txt
cp .env.example .env
```

### 3. Execute o pipeline completo
```bash
make run-pipeline    # Bronze → Silver → Gold
make train           # Treina e avalia os modelos de ML
make run-dashboard   # Inicia o dashboard em http://localhost:8501
```

### 4. Ou execute passo a passo
```bash
# Ingestão (Bronze)
python src/ingestion/ingest_kaggle.py
python src/ingestion/ingest_bcb_api.py

# Transformação (Silver → Gold)
python src/transformation/silver_transform.py
python src/transformation/gold_transform.py

# Machine Learning
python src/models/train_model.py
python src/models/evaluate_model.py
```

### 5. Testes
```bash
make test
# ou: pytest tests/ -v --cov=src
```

---

## 📁 Estrutura do Projeto

```
credit-risk-data-pipeline/
│
├── 📖 README.md
├── 📦 requirements.txt
├── ⚙️  Makefile                    ← atalhos: make run-pipeline, make train
├── 🔧 .env.example
│
├── 📂 data/
│   ├── bronze/                    ← dados brutos (CSV originais)
│   ├── silver/                    ← dados limpos e validados
│   └── gold/                      ← tabelas dimensionais + agregações
│
├── 📂 src/
│   ├── ingestion/
│   │   ├── ingest_kaggle.py       ← camada Bronze (dataset de crédito)
│   │   └── ingest_bcb_api.py      ← camada Bronze (Taxa Selic — API BCB)
│   ├── transformation/
│   │   ├── silver_transform.py    ← limpeza, validação, tipagem
│   │   └── gold_transform.py      ← modelagem dimensional (Star Schema)
│   ├── models/
│   │   ├── feature_engineering.py ← encoding, SMOTE, scaler
│   │   ├── train_model.py         ← treina LR, RF, XGBoost
│   │   └── evaluate_model.py      ← ROC, matriz confusão, SHAP
│   ├── dashboard/
│   │   └── app.py                 ← Streamlit + Plotly
│   └── utils/
│       ├── db.py                  ← DuckDB + PostgreSQL
│       └── logger.py              ← logging padronizado (Loguru)
│
├── 📂 sql/
│   ├── silver/01_silver_transformations.sql
│   └── gold/02_gold_analytics.sql
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb               ← Análise Exploratória
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Evaluation.ipynb
│
├── 📂 tests/
│   ├── test_ingestion.py
│   └── test_transformation.py
│
├── 📂 .github/workflows/
│   └── ci.yml                     ← CI/CD (Black + Flake8 + pytest)
│
└── 📂 docs/
    ├── decisions.md               ← ADR: registro de decisões técnicas
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

---

## 🗓️ Sprint Log (Mindset Ágil)

| Sprint | Entrega | Status |
|---|---|---|
| Sprint 1 | Estrutura do projeto + ingestão Bronze | ✅ Concluído |
| Sprint 2 | Transformação Silver (limpeza + validação) | ✅ Concluído |
| Sprint 3 | Gold Layer (Star Schema + agregações SQL) | ✅ Concluído |
| Sprint 4 | Machine Learning (Feature Eng. + XGBoost) | ✅ Concluído |
| Sprint 5 | Dashboard Streamlit + CI/CD | ✅ Concluído |
| Sprint 6 | Testes unitários + documentação final | ✅ Concluído |

---

## 📚 Dataset

**Kaggle Credit Risk Dataset**
- 32.581 registros de empréstimos
- 12 features (demográficas, financeiras, históricas)
- Target binário: `loan_status` (0 = adimplente, 1 = inadimplente)
- Taxa de inadimplência natural: ~21.8%

**API Banco Central do Brasil**
- Série 432: Taxa Selic Over (diária)
- Contexto macroeconômico para análise de risco sistêmico

---

## 🔮 Próximos Passos

- [ ] Deploy no Streamlit Community Cloud (link público)
- [ ] Migração para Azure Data Factory / AWS Glue
- [ ] Implementação de Apache Airflow para orquestração
- [ ] Modelo de NLP para análise de reclamações de crédito
- [ ] Monitoramento de data drift com Evidently AI

---

## 👩‍💻 Sobre a Autora

**Nayane Araújo** — Pós-Graduanda em Engenharia de Dados

[![GitHub](https://img.shields.io/badge/GitHub-Nayanearaujo-181717?logo=github&logoColor=white)](https://github.com/Nayanearaujo)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-0077B5?logo=linkedin&logoColor=white)](https://linkedin.com/in/nayanearaujo)

---

<div align="center">

**Arquitetura Medallion · Python · SQL · XGBoost · Streamlit · GitHub Actions**

*Projeto de portfólio — Pós-Graduação em Engenharia de Dados*

</div>
