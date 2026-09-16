"""
Scripts Dev - Gerador de Dados Sintéticos para Desenvolvimento Offline
========================================================================
AVISO: Este script gera dados SINTÉTICOS puramente para desenvolvimento
e testes locais offline (sem acesso à internet).
NÃO é a fonte oficial dos dados nem dos resultados analíticos do README.
Para os dados reais, execute a ingestão oficial:
    python src/ingestion/ingest_openml.py

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)
n = 32581
grades = ["A", "B", "C", "D", "E", "F", "G"]
grade_weights = [0.28, 0.22, 0.20, 0.14, 0.08, 0.05, 0.03]
grade_col = np.random.choice(grades, n, p=grade_weights)
rate_map = {"A": 7.5, "B": 10.5, "C": 13.5, "D": 16.5, "E": 19.5, "F": 22.5, "G": 25.0}
base_rates = np.array([rate_map[g] for g in grade_col])

df = pd.DataFrame(
    {
        "person_age": np.clip(np.random.normal(27, 6, n).astype(int), 18, 80),
        "person_income": np.clip(
            np.random.lognormal(10.8, 0.7, n), 4000, 6000000
        ).astype(int),
        "person_home_ownership": np.random.choice(
            ["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[0.50, 0.41, 0.08, 0.01]
        ),
        "person_emp_length": np.where(
            np.random.rand(n) < 0.05,
            np.nan,
            np.clip(np.random.exponential(4, n), 0, 41).round(1),
        ),
        "loan_intent": np.random.choice(
            [
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION",
            ],
            n,
            p=[0.20, 0.19, 0.17, 0.15, 0.15, 0.14],
        ),
        "loan_grade": grade_col,
        "loan_amnt": np.random.choice(
            [2000, 3500, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 35000], n
        ),
        "loan_int_rate": np.where(
            np.random.rand(n) < 0.10,
            np.nan,
            np.clip(base_rates + np.random.normal(0, 1.5, n), 5.0, 30.0).round(2),
        ),
        "loan_percent_income": np.clip(np.random.beta(2, 6, n), 0.01, 0.99).round(2),
        "cb_person_default_on_file": np.random.choice(["N", "Y"], n, p=[0.83, 0.17]),
        "cb_person_cred_hist_length": np.clip(np.random.poisson(5, n), 1, 30),
    }
)
default_prob = (
    0.05
    + (pd.Categorical(grade_col, categories=grades).codes * 0.04)
    + (df["loan_percent_income"] * 0.15)
    + ((df["cb_person_default_on_file"] == "Y").astype(float) * 0.10)
)
df["loan_status"] = (np.random.rand(n) < default_prob.clip(0, 1)).astype(int)

out = Path("data/bronze/credit_risk_raw.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Gerado: {len(df):,} linhas x {len(df.columns)} colunas (SINTETICO)")
print(f"Inadimplencia: {df['loan_status'].mean()*100:.1f}%")
print(f"Salvo em: {out}")
