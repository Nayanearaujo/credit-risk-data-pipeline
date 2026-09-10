-- =============================================================================
-- Gold Layer — Queries de Modelagem Dimensional e Análise de Negócio
-- =============================================================================
-- Projeto: Pipeline de Risco de Crédito
-- Autora:  Nayane Araújo | github.com/Nayanearaujo
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Criação da tabela fato: fact_loans (Star Schema)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_loans AS
SELECT
    b.borrower_id,
    lt.loan_type_id,
    s.loan_amnt,
    s.loan_int_rate,
    s.loan_percent_income,
    s.loan_status,
    CASE WHEN s.loan_status = 1 THEN 'Inadimplente' ELSE 'Adimplente' END AS loan_status_label,
    -- Métricas derivadas
    s.loan_amnt * s.loan_int_rate / 100                                    AS annual_interest_cost,
    s.loan_amnt * (1 + s.loan_int_rate / 100)                             AS estimated_total_cost
FROM silver_credit_risk s
LEFT JOIN dim_borrower b
    ON s.person_age = b.person_age AND s.person_income = b.person_income
LEFT JOIN dim_loan_type lt
    ON s.loan_intent = lt.loan_intent AND s.loan_grade = lt.loan_grade;


-- -----------------------------------------------------------------------------
-- 2. Análise: Risco por Grade — responde "Qual grade tem maior inadimplência?"
-- -----------------------------------------------------------------------------
SELECT
    lt.loan_grade,
    lt.risk_level,
    COUNT(f.loan_amnt)                            AS total_emprestimos,
    SUM(f.loan_status)                            AS inadimplentes,
    ROUND(AVG(f.loan_status) * 100, 2)            AS taxa_inadimplencia_pct,
    ROUND(SUM(f.loan_amnt), 2)                    AS volume_total,
    ROUND(AVG(f.loan_int_rate), 2)                AS taxa_juros_media
FROM fact_loans f
JOIN dim_loan_type lt ON f.loan_type_id = lt.loan_type_id
GROUP BY lt.loan_grade, lt.risk_level
ORDER BY taxa_inadimplencia_pct DESC;


-- -----------------------------------------------------------------------------
-- 3. Análise: Perfil dos Inadimplentes vs. Adimplentes
-- -----------------------------------------------------------------------------
SELECT
    f.loan_status_label,
    ROUND(AVG(b.person_age), 1)                   AS media_idade,
    ROUND(AVG(b.person_income), 0)                AS media_renda,
    ROUND(AVG(b.person_emp_length), 1)            AS media_anos_emprego,
    ROUND(AVG(f.loan_amnt), 0)                    AS media_valor_emprestimo,
    ROUND(AVG(f.loan_int_rate), 2)                AS media_taxa_juros,
    ROUND(AVG(f.loan_percent_income) * 100, 2)    AS media_comprometimento_renda_pct,
    COUNT(*)                                       AS total
FROM fact_loans f
JOIN dim_borrower b ON f.borrower_id = b.borrower_id
GROUP BY f.loan_status_label;


-- -----------------------------------------------------------------------------
-- 4. Análise: Risco por Faixa Etária
-- -----------------------------------------------------------------------------
SELECT
    b.age_group,
    COUNT(*)                                       AS total_emprestimos,
    SUM(f.loan_status)                             AS inadimplentes,
    ROUND(AVG(f.loan_status) * 100, 2)             AS taxa_inadimplencia_pct,
    ROUND(AVG(b.person_income), 0)                 AS media_renda,
    ROUND(AVG(f.loan_amnt), 0)                     AS media_valor_solicitado
FROM fact_loans f
JOIN dim_borrower b ON f.borrower_id = b.borrower_id
GROUP BY b.age_group
ORDER BY b.age_group;


-- -----------------------------------------------------------------------------
-- 5. Top 10 piores tomadores por risco combinado
-- -----------------------------------------------------------------------------
WITH risk_score AS (
    SELECT
        f.borrower_id,
        b.person_age,
        b.person_income,
        f.loan_amnt,
        f.loan_int_rate,
        f.loan_percent_income,
        b.has_prior_default,
        -- Score de risco composto (simplificado)
        (f.loan_percent_income * 0.4
         + (f.loan_int_rate / 25.0) * 0.3
         + b.has_prior_default * 0.3)             AS composite_risk_score
    FROM fact_loans f
    JOIN dim_borrower b ON f.borrower_id = b.borrower_id
    WHERE f.loan_status = 0  -- apenas adimplentes (candidatos futuros)
)
SELECT *
FROM risk_score
ORDER BY composite_risk_score DESC
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 6. Relatório executivo: sumário de KPIs para o dashboard
-- -----------------------------------------------------------------------------
SELECT
    COUNT(*)                                           AS total_emprestimos,
    SUM(loan_status)                                   AS total_inadimplentes,
    ROUND(AVG(loan_status) * 100, 2)                   AS taxa_inadimplencia_pct,
    ROUND(SUM(loan_amnt), 0)                           AS volume_total_emprestado,
    ROUND(AVG(loan_amnt), 0)                           AS ticket_medio,
    ROUND(AVG(loan_int_rate), 2)                       AS taxa_juros_media,
    ROUND(SUM(annual_interest_cost), 0)                AS receita_juros_estimada
FROM fact_loans;
