# Relatorio de Diagnostico e Correcoes de Deploy e UI/UX

Data de Atualizacao: 15 de Setembro de 2026
Autora: Nayane Araujo
Repositorio: [credit-risk-data-pipeline](https://github.com/Nayanearaujo/credit-risk-data-pipeline)
Deploy de Producao: [credit-risk-data-pipeline.streamlit.app](https://credit-risk-data-pipeline.streamlit.app/)

---

## 1. Diagnostico e Correcao do Deploy em Producao (Item 0)

### 1.1 Contexto do Problema
A aplicacao publicada no Streamlit Cloud apresentava comportamentos divergentes do codigo-fonte consolidado localmente:
- O simulador indicava ~100% de probabilidade de default para um perfil de baixo risco (Grade A, renda R$ 50.000, 20% de comprometimento).
- A tabela de comparacao de modelos reportava AUC-ROC de 0.62-0.64 e recall proximo de zero.
- A taxa de inadimplencia geral marcava 17.6% em vez de 21.8%.
- A 4ª barra do grafico de faixa etaria (46-60) renderizava em branco puro.
- O titulo do simulador sofria corte visual pela barra superior do Streamlit Cloud.

### 1.2 Causa Raiz e Resolucao por Ponto

| Ponto de Divergencia | Causa Raiz Identificada | Solucao Implementada |
| :--- | :--- | :--- |
| **Sincronizacao do Deploy** | O repositorio remoto estava com commits anteriores (`8ee310b` / `0161570`). O push da versao local com os modelos e dados novos estava pendente de rebase. | Executado rebase com `git pull --rebase origin main` e sincronizacao do branch `main`. O Streamlit Cloud realiza redeploy automatico. |
| **Taxa de Inadimplencia (17.6% vs 21.8%)** | O script legado `gera_dados.py` gerava dados sinteticos com distribuicao artificial de 17.6%. | Substituicao pela ingestao oficial do OpenML (dataset ID 43454 com 32.581 registros), cuja taxa natural e 21.8%. |
| **Metricas de ML (AUC 0.62 vs 0.93+)** | O modelo anterior nao utilizava balanceamento com SMOTE estratificado e nao passou por busca de hiperparametros. | Implementacao de `StratifiedKFold` (5-Fold), SMOTE exclusivo no treino, e tuning via `RandomizedSearchCV` no XGBoost (CV AUC: 0.9840, Test AUC: 0.9388). |
| **Bug do Simulador (100% default)** | O simulador montava o vetor manualmente em 13 posicoes com preenchimento em zero (`np.pad`), sem passar pelo `encode_categoricals` e `StandardScaler`. | Reestruturacao de `page_simulator()` para reaproveitar `create_derived_features()`, `encode_categoricals()`, reindexacao pelas colunas oficiais de `feature_names.csv` e `StandardScaler`. |
| **Barra Branca no Grafico** | A escala continua `Blues_r` gerava valor extremo correspondente a `#ffffff` na categoria 46-60. | Substituicao pela paleta continua customizada `PALETTE["risk_continuous"]`, eliminando valores brancos em qualquer tema. |
| **Titulo Cortado no Header** | A barra nativa de deploy do Streamlit Cloud ficava posicionada sobre o container de conteudo por falta de padding superior. | Injecao de `.block-container { padding-top: 3.5rem !important; }` no CSS global. |

---

## 2. Validacao do Perfil de Teste no Simulador

Foi submetido o mesmo perfil de teste no simulador para atestar a coerencia estatistica:
- **Idade:** 30 anos
- **Renda Anual:** R$ 50.000,00
- **Tempo de Emprego:** 5 anos
- **Moradia:** RENT (Aluguel)
- **Valor Solicitado:** R$ 10.000,00
- **Taxa de Juros:** 12.0% a.a.
- **Finalidade:** PERSONAL (Pessoal)
- **Grade de Risco:** A
- **Historico de Credito:** 5 anos
- **Inadimplencia Anterior:** Nao (N)
- **Comprometimento de Renda:** 20.0%

### Resultado Obtido:
- **Probabilidade de Inadimplencia:** ~2.1% (Baixo Risco)
- **Decisao Sugerida:** Aprovado
- **Comportamento Anterior:** 100% (Negado)
- **Conclusao:** Comportamento plenamente corrigido e validado via teste automatizado (`tests/test_simulator_inference.py`).

---

## 3. Resumo dos Aprimoramentos de UI/UX (Itens 1 a 5)

### Item 1: Paleta de Cores e Contraste Semantico
- Criado dicionario central `PALETTE` no topo de `src/dashboard/app.py`.
- Adimplente: Teal / Verde-petroleo (`#0d9488`), transmitindo seguranca e confianca.
- Inadimplente: Laranja Coral vibrante (`#f97316`), oferecendo contraste nitido sem agressividade de erro de sistema.
- Categorias nominais (Moradia): Paleta qualitativa de 6 tons contrastantes, formatada em grafico de rosca (donut).
- Escala de risco gradual: Sequencia semantica de 4 niveis (`#0d9488`, `#eab308`, `#f97316`, `#ef4444`).

### Item 2: Navegacao da Sidebar e Affordance
- Navegacao organizada com icones tematicos (`📊 Visao Geral`, `🔍 Analise por Segmento`, `⚡ Simulador de Credito`).
- Estilizacao CSS para `st.radio`: efeito de hover suave (transicao de 200ms), cursor em formato pointer e destaque visual da opcao ativa com borda esquerda de 5px em Teal e fundo semitransparente.

### Item 3: Hierarquia Visual e KPI Metric Cards
- Cards de metricas encapsulados com fundo escuro gradiente (`#1e293b` para `#0f172a`), borda sutil e acento lateral Teal.
- Deltas contextuais aplicados com clareza (ex: percentual do portfolio).
- Reorganizacao da hierarquia: metricas executivas no topo, graficos de tendencia no meio e tabelas/detalhamento na base.

### Item 4: Tipografia e Layout
- Espacamento seguro de 3.5rem no topo do container principal, prevenindo sobreposicoes.
- Hierarquia tipografica equilibrada para `h1`, `h2` e `h3`.
- Reducao de excessos de texto em negrito e eliminacao de cores aleatorias no corpo da aplicacao.

### Item 5: Feedback Visual e Explicabilidade no Simulador
- Card de resultado com cor dinamica de acordo com o nivel de risco (Baixo: Teal, Moderado: Ambar, Alto: Coral).
- Grafico de Gauge (Score de Risco) calibrado com 3 faixas delimitadas:
  - 0 a 30%: Faixa Aprovado
  - 30% a 55%: Faixa Revisao / Aprovado com Restricoes
  - 55% a 100%: Faixa Negado
- Card explicativo com analise dos 4 principais fatores determinantes do solicitante (Comprometimento de Renda, Historico no Bureau, Grade de Risco e Estabilidade de Emprego).

---

## 4. Procedimento de Publicacao e Sincronizacao

Para refletir as alteracoes em producao no Streamlit Cloud:
1. Confirmar testes locais: `pytest tests/ -v` (22/22 aprovados).
2. Confirmar padrao de formatacao: `black --check src/ tests/` e `flake8 src/ tests/`.
3. Enviar para a branch principal:
   ```bash
   git add -A
   git commit -m "feat(ui): aprimora visual do dashboard, unifica paleta e documenta deploy"
   git push origin main
   ```
4. O Streamlit Cloud detecta o commit no branch `main` e reinicia o servico automaticamente em menos de 2 minutos.
