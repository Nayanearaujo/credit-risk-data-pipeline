# 📋 Sprint Log — Pipeline de Risco de Crédito
# Autora: Nayane Araújo | Mindset Ágil aplicado ao portfólio

## Sprint 1 — Base do Projeto
**Período:** Semana 1 | **Status:** ✅ Concluído

### Entregues
- [x] Estrutura de pastas seguindo padrão de mercado
- [x] requirements.txt com stack completa
- [x] .env.example e .gitignore
- [x] Makefile com atalhos de automação
- [x] Módulos utilitários: db.py, logger.py
- [x] Script de ingestão Bronze (Kaggle + API BCB)

### Decisões tomadas
- Escolhido DuckDB como banco analítico local (ver ADR-002)
- Estrutura de pastas preparada para migração a Cloud

---

## Sprint 2 — Camada Silver (ETL)
**Período:** Semana 1 | **Status:** ✅ Concluído

### Entregues
- [x] Remoção de duplicatas com log de quantidade
- [x] Tipagem correta de todas as colunas
- [x] Tratamento de nulos com estratégia documentada (mediana por grade)
- [x] Validação de ranges de negócio (idade 18-100, renda > 0)
- [x] Flags de qualidade e rastreabilidade (_source, _silver_timestamp)
- [x] Queries SQL Silver documentadas
- [x] Integração com DuckDB

### Aprendizados
- Nulos na taxa de juros: melhor usar mediana por grade do que mediana global
  (taxas variam muito entre grades A e G)

---

## Sprint 3 — Camada Gold (Modelagem Dimensional)
**Período:** Semana 1 | **Status:** ✅ Concluído

### Entregues
- [x] Star Schema: fact_loans, dim_borrower, dim_loan_type
- [x] Features derivadas: age_group, income_band, has_prior_default
- [x] 3 agregações de negócio prontas para dashboard
- [x] Queries Gold com 6 análises documentadas
- [x] Relatório executivo SQL (KPIs para o dashboard)

---

## Sprint 4 — Machine Learning
**Período:** Semana 2 | **Status:** ✅ Concluído

### Entregues
- [x] Feature Engineering: derived features, OrdinalEncoder, One-Hot, SMOTE
- [x] 3 modelos treinados: Regressão Logística, Random Forest, XGBoost
- [x] Métricas: AUC-ROC, F1, Precision, Recall, KS Statistic
- [x] Avaliação visual: ROC curve, confusion matrix, feature importance
- [x] Scaler e model salvos em .pkl
- [x] Tabela de comparação Gold: model_comparison.csv

### Decisões tomadas
- SMOTE aplicado apenas no treino (ver ADR-003)
- XGBoost eleito modelo de produção (ver ADR-004)

---

## Sprint 5 — Dashboard e CI/CD
**Período:** Semana 2 | **Status:** ✅ Concluído

### Entregues
- [x] Dashboard Streamlit com 3 páginas: Visão Geral, Análise, Simulador
- [x] Simulador de crédito com gauge chart
- [x] GitHub Actions: lint (Black + Flake8) + pytest automático
- [x] README.md profissional com arquitetura e badges

---

## Sprint 6 — Qualidade e Documentação
**Período:** Semana 2 | **Status:** ✅ Concluído

### Entregues
- [x] Testes unitários: ingestion e transformation (pytest)
- [x] docs/decisions.md com 5 ADRs documentados
- [x] sprint_log.md (este arquivo)
- [x] Código revisado com docstrings completos

---

## 📌 Backlog (Próximas versões)

- [ ] Notebooks Jupyter (EDA, Feature Eng., Model Evaluation)
- [ ] Deploy Streamlit Community Cloud
- [ ] Migração para Azure Data Factory
- [ ] Monitoramento de data drift (Evidently AI)
- [ ] Módulo NLP (análise de reclamações)
- [ ] Apache Airflow para orquestração
