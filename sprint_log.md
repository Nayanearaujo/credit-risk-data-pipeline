# Diário de Desenvolvimento (Sprints)

Autora: Nayane Araújo | github.com/Nayanearaujo

Este registro acompanha o desenvolvimento prático do pipeline, documentando decisões, erros encontrados no caminho e correções aplicadas.

---

## Sprint 1: Estrutura inicial e conexões

Semana 1. Status: Concluído.

Comecei desenhando a estrutura de diretórios do projeto para separar bem as responsabilidades de ingestão, transformação e modelagem. Para o banco analítico de desenvolvimento local, escolhi o DuckDB (registrado na ADR-001) porque ele roda in-process sem precisar subir container PostgreSQL logo no primeiro dia, além de conversar direto com DataFrames do Pandas. Criei o Makefile para facilitar comandos recorrentes, configurei o logger com Loguru e estruturei a ingestão da API de taxas Selic do Banco Central.

Entregas:
- Estrutura de pastas do pipeline (src, data, sql, tests, docs)
- requirements.txt e Makefile
- Conectores de banco em db.py e configuração do logger
- Script de ingestão da série 432 do Banco Central (Taxa Selic)

---

## Sprint 2: Camada Silver e regras de limpeza

Semana 1. Status: Concluído.

Na camada Silver, o objetivo foi transformar os dados brutos em uma base confiável e tipada. Ao inspecionar os nulos da taxa de juros (`loan_int_rate`), percebi que preencher com a média global distorceria o risco, já que a taxa cobrada depende diretamente do rating de crédito do cliente. A decisão técnica foi aplicar mediana agrupada por `loan_grade`, preservando o perfil de risco original.

O que foi implementado:
- Deduplicação com contagem de registros removidos (165 duplicatas no dataset original)
- Conversão e tipagem estrita de colunas numéricas e strings padronizadas
- Tratamento de nulos com estratégia diferenciada para tempo de emprego e taxa de juros
- Validação de limites aceitáveis (idade entre 18 e 100 anos, renda positiva, percentual de comprometimento até 1.0)
- Criação de metadados de auditoria (`_silver_timestamp`, `_source`, `_row_id`) salvos em DuckDB e CSV

---

## Sprint 3: Modelagem dimensional Gold

Semana 1. Status: Concluído.

Para estruturar os dados para análise e consumo no dashboard, modelei um Star Schema contendo a tabela fato `fact_loans` e as dimensões `dim_borrower` e `dim_loan_type`. Também escrevi agregações focadas em responder perguntas práticas de crédito: taxa de default por faixa de risco (grade), por finalidade do empréstimo e por faixa etária. Todas as tabelas foram gravadas na camada Gold em CSV e sincronizadas no banco analítico.

---

## Sprint 4: Modelagem de Machine Learning e calibração

Semana 2. Status: Concluído.

Nesta etapa montei o pipeline preditivo de inadimplência, mas nem tudo correu de primeira. Durante a revisão técnica do projeto, descobri um bug crítico que estava no simulador do dashboard: o vetor de entrada estava sendo montado manualmente com 13 valores em ordem arbitrária, sem passar pelo StandardScaler salvo no treino e ignorando as colunas geradas pelo One-Hot Encoding de finalidade e moradia. O código antigo mascarava esse erro fazendo um preenchimento cego com zeros (`np.pad`) só para bater a quantidade de colunas exigida pelo modelo, o que gerava probabilidades sem sentido na interface.

Para resolver, refatorei o fluxo de inferência para importar e reaproveitar exatamente as funções de produção `create_derived_features()` e `encode_categoricals()` de `feature_engineering.py`. Agora o input do usuário gera um DataFrame de uma linha com os mesmos nomes de colunas crus, é transformado pela mesma lógica do treino, reindexado para a lista oficial de `feature_names.csv` e escalonado antes da predição.

Além disso, troquei a avaliação em split único por validação cruzada estratificada em 5 folds e adicionei busca de hiperparâmetros (RandomizedSearchCV) para o XGBoost. Também incluí a análise de calibração das probabilidades usando Brier Score e curva de confiabilidade, já que em crédito prever uma probabilidade fiel à realidade é tão importante quanto o ordenamento do ranking.

Resultados consolidados:
- XGBoost Tuned: AUC-ROC médio em CV de 0.9834 (±0.0007) e 0.9485 no teste independente
- Brier Score de 0.0547 (0.0542 após calibração isotônica/sigmoide)
- Melhores parâmetros salvos em `data/gold/best_params.json` (ADR-006)

---

## Sprint 5: Dashboard interativo

Semana 2. Status: Concluído.

Desenvolvi o dashboard no Streamlit dividido em três áreas: Visão Geral com métricas de carteira e volume total, Análise por Segmento para filtros dinâmicos e o Simulador de Crédito em tempo real. Com a correção feita no alinhamento das features no Sprint 4, o simulador agora reflete com precisão o impacto das variáveis, permitindo acompanhar o aumento consistente da probabilidade de inadimplência conforme a nota de risco piora de A até G.

---

## Sprint 6: Testes automatizados e CI

Semana 2. Status: Concluído.

Outro ponto que precisei corrigir com rigor: quando olhei os primeiros arquivos de teste unitário, percebi que `test_transformation.py` e `test_ingestion.py` tinham reescrito cópias das funções de limpeza dentro do próprio teste, em vez de importar o código real da pasta `src/`. Isso significava que se uma função de produção quebrasse, o teste continuaria passando verde. Removi toda essa duplicação e passei a importar as funções reais de `src.transformation.silver_transform` e `src.ingestion.ingest_openml`.

Também criei um teste de integração de ponta a ponta (`tests/test_integration.py`) com uma fixture sintética controlada de 200 linhas para verificar o encadeamento Silver -> Gold -> Features -> Modelo rápido. No GitHub Actions, configurei a verificação de código com Black e Flake8 cobrindo tanto `src/` quanto `tests/`, garantindo que qualquer push passe por validação real antes do deploy.

---

## Backlog e melhorias planejadas

- Deploy público contínuo do Streamlit Community Cloud
- Implementação de DAG no Apache Airflow para agendamento diário
- Monitoramento de drift de dados e conceito usando Evidently AI
- Módulo de análise textual para reclamações de crédito
