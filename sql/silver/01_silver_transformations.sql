-- =============================================================================
-- Silver Layer - Queries de Transformação e Validação
-- =============================================================================
-- Projeto: Pipeline de Risco de Crédito
-- Autora:  Nayane Araújo | github.com/Nayanearaujo
-- Banco:   DuckDB / PostgreSQL / SQL Server (sintaxe compatível)
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Verificação de qualidade: contagem de nulos por coluna
-- -----------------------------------------------------------------------------
-- Use para auditar a camada Bronze antes de limpar.
SELECT
    COUNT(*) AS total_registros,
    SUM(CASE WHEN person_age IS NULL THEN 1 ELSE 0 END)           AS nulos_idade,
    SUM(CASE WHEN person_income IS NULL THEN 1 ELSE 0 END)        AS nulos_renda,
    SUM(CASE WHEN person_emp_length IS NULL THEN 1 ELSE 0 END)    AS nulos_tempo_emprego,
    SUM(CASE WHEN loan_int_rate IS NULL THEN 1 ELSE 0 END)        AS nulos_taxa_juros,
    SUM(CASE WHEN loan_amnt IS NULL THEN 1 ELSE 0 END)            AS nulos_valor_emprestimo
FROM bronze_credit_risk;


-- -----------------------------------------------------------------------------
-- 2. Limpeza: remove duplicatas (CTE + ROW_NUMBER)
-- -----------------------------------------------------------------------------
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY person_age, person_income, loan_amnt, loan_int_rate
            ORDER BY ROWID
        ) AS rn
    FROM bronze_credit_risk
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;


-- -----------------------------------------------------------------------------
-- 3. Tratamento de nulos: taxa de juros → mediana por grade
-- -----------------------------------------------------------------------------
WITH medians AS (
    SELECT
        loan_grade,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY loan_int_rate) AS median_rate
    FROM bronze_credit_risk
    WHERE loan_int_rate IS NOT NULL
    GROUP BY loan_grade
)
SELECT
    b.*,
    COALESCE(b.loan_int_rate, m.median_rate) AS loan_int_rate_clean
FROM bronze_credit_risk b
LEFT JOIN medians m ON b.loan_grade = m.loan_grade;


-- -----------------------------------------------------------------------------
-- 4. Validação de ranges de negócio
-- -----------------------------------------------------------------------------
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN person_age < 18 OR person_age > 100 THEN 1 ELSE 0 END) AS idades_invalidas,
    SUM(CASE WHEN person_income <= 0 THEN 1 ELSE 0 END)                   AS rendas_invalidas,
    SUM(CASE WHEN person_emp_length > 60 THEN 1 ELSE 0 END)               AS emprego_invalido,
    SUM(CASE WHEN loan_percent_income > 1 THEN 1 ELSE 0 END)              AS renda_comprometida_invalida
FROM bronze_credit_risk;


-- -----------------------------------------------------------------------------
-- 5. Distribuição de inadimplência por grade (validação rápida Silver)
-- -----------------------------------------------------------------------------
SELECT
    loan_grade,
    COUNT(*)                                      AS total,
    SUM(loan_status)                              AS inadimplentes,
    ROUND(AVG(loan_status) * 100, 2)              AS taxa_inadimplencia_pct,
    ROUND(AVG(loan_int_rate), 2)                  AS taxa_juros_media
FROM silver_credit_risk
GROUP BY loan_grade
ORDER BY loan_grade;
