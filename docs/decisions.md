# Registro de Decisões de Arquitetura (ADR)
# ==========================================
# Projeto: Pipeline de Risco de Crédito
# Autora:  Nayane Araújo | github.com/Nayanearaujo

Este documento registra as principais decisões técnicas tomadas durante o projeto
e o raciocínio por trás de cada uma. É uma prática de **engenharia de software**
que diferencia um projeto de portfólio dos demais.

---

## ADR-001: Escolha do DuckDB como banco analítico padrão

**Data:** Setembro/2026
**Status:** Aprovado

**Contexto:**
Precisava de um banco para executar queries analíticas sem depender de uma
infraestrutura PostgreSQL em produção local.

**Decisão:**
Adotar o DuckDB como banco analítico padrão para desenvolvimento e testes locais,
mantendo suporte opcional a PostgreSQL para ambiente de produção.

**Razões:**
- DuckDB executa queries SQL analíticas in-process, sem servidor
- Performance superior ao Pandas para aggregações em datasets maiores
- API compatível com pandas (`.df()`)
- Fácil migração para Databricks (Delta Lake) ou AWS Athena em produção

**Trade-offs:**
- Não suporta transações OLTP complexas
- Não é adequado para muitos usuários simultâneos (OLAP apenas)

---

## ADR-002: Arquitetura Medallion (Bronze / Silver / Gold)

**Data:** Setembro/2026
**Status:** Aprovado

**Contexto:**
Precisava estruturar o pipeline de dados de forma que cada camada tivesse
responsabilidade clara e fosse rastreável.

**Decisão:**
Implementar a Arquitetura Medallion com três camadas:
- **Bronze:** dados brutos sem alterações (audit trail)
- **Silver:** dados limpos, validados e tipados
- **Gold:** dados prontos para negócio (dimensional, agregações)

**Razões:**
- Padrão de mercado no Databricks/Azure e AWS
- Permite reprocessamento de qualquer camada sem perda de dados
- Facilita a colaboração entre Engenharia e Ciência de Dados
- Auditabilidade completa do dado

---

## ADR-003: SMOTE para balanceamento de classes

**Data:** Setembro/2026
**Status:** Aprovado

**Contexto:**
O dataset tem desbalanceamento: ~78% adimplentes vs ~22% inadimplentes.
Treinar sem balancear gera um modelo tendencioso para a classe majoritária.

**Decisão:**
Aplicar SMOTE (Synthetic Minority Over-sampling Technique) **apenas no conjunto
de treino**, nunca no teste. Scaler também é ajustado apenas no treino.

**Razões:**
- SMOTE gera exemplos sintéticos realistas para a classe minoritária
- Aplicar no teste causaria data leakage (contaminaria a avaliação)
- Alternativas consideradas: under-sampling (perde dados), class_weight (menos eficaz)

---

## ADR-004: XGBoost como modelo principal

**Data:** Setembro/2026
**Status:** Aprovado

**Contexto:**
Testei três modelos: Regressão Logística, Random Forest e XGBoost.

**Decisão:**
Usar XGBoost como modelo de produção, mantendo a Regressão Logística como
baseline interpretável para explicabilidade ao negócio.

**Razões:**
- XGBoost superou os demais em AUC-ROC e F1-Score
- Gradient boosting é estado da arte para tabular data
- Regressão Logística preservada: mais fácil de explicar ao cliente

---

## ADR-005: Streamlit para dashboard (vs. Power BI / Tableau)

**Data:** Setembro/2026
**Status:** Aprovado

**Contexto:**
Precisava de uma forma de visualizar resultados que fosse:
1. Executável localmente sem licença
2. Integrável com o modelo de ML em Python
3. Impressionante no portfólio GitHub

**Decisão:**
Usar Streamlit para o dashboard interativo.

**Razões:**
- Código Python puro — demonstra skills de programação
- Integra nativamente com Scikit-Learn e XGBoost
- Deploy gratuito no Streamlit Community Cloud
- Plotly para gráficos interativos

**Trade-offs:**
- Power BI seria preferível em ambiente corporativo com dados não-técnicos
- Streamlit não substitui BI enterprise, mas é ideal para portfólio técnico
