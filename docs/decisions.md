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
- Código Python puro - demonstra skills de programação
- Integra nativamente com Scikit-Learn e XGBoost
- Deploy gratuito no Streamlit Community Cloud
- Plotly para gráficos interativos

**Trade-offs:**
- Power BI seria preferível em ambiente corporativo com dados não-técnicos
- Streamlit não substitui BI enterprise, mas é ideal para portfólio técnico

---

## ADR-006: Validação Cruzada Estratificada, Tuning do XGBoost e Calibração

**Data:** Setembro/2026
**Status:** Aprovado

**Contexto:**
Avaliar modelos apenas em um split único de treino/teste introduz variância e risco de sobreajuste aos dados de teste. Além disso, em concessão de crédito, uma probabilidade mal calibrada (ex: modelo prevendo 80% quando a inadimplência real é 40%) distorce políticas de crédito e precificação de taxa.

**Decisão:**
1. Adotar validação cruzada estratificada em 5 folds (StratifiedKFold) para estimar média e desvio padrão do AUC-ROC.
2. Implementar busca aleatória de hiperparâmetros (RandomizedSearchCV) para o XGBoost, persistindo os parâmetros vencedores em `data/gold/best_params.json`.
3. Adicionar o Brier Score e reliability curves para auditar a calibração das probabilidades previstas.

**Razões:**
- CV 5-fold assegura estabilidade da estimativa de generalização (AUC médio = 0.9834 ± 0.0007 no treino balanceado; 0.9485 no teste independente).
- Tuning via RandomizedSearchCV otimizou max_depth (6), learning_rate (0.08) e subsample (0.95), gerando ganhos sem onerar o tempo de pipeline.
- Brier score de 0.0547 confirma excelente alinhamento probabilístico entre predições e inadimplência real observada.

