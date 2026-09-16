"""
run_pipeline.py - Roda o pipeline completo de uma vez
======================================================
Bronze -> Silver -> Gold -> Modelo -> Dashboard pronto!

Execute com:
    python run_pipeline.py

Nao precisa de DuckDB, PostgreSQL nem nenhuma outra dependencia especial.
So pandas, numpy e scikit-learn (que o Anaconda ja tem instalado).

Autora: Nayane Araujo | github.com/Nayanearaujo
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
BRONZE = ROOT / "data" / "bronze" / "credit_risk_raw.csv"
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR = ROOT / "data" / "gold"
MODELS_DIR = ROOT / "src" / "models"
DOCS_DIR = ROOT / "docs"

for d in [SILVER_DIR, GOLD_DIR, MODELS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def step1_silver():
    """Bronze -> Silver: limpeza e tipagem."""
    print("\n[1/4] SILVER - Limpeza dos dados...")
    df = pd.read_csv(BRONZE)
    print(f"  Bronze: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

    # Remove duplicatas
    antes = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicatas removidas: {antes - len(df)}")

    # Padroniza strings
    str_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # Trata nulos
    if "person_emp_length" in df.columns:
        med = df["person_emp_length"].median()
        df["person_emp_length"] = df["person_emp_length"].fillna(med)

    if "loan_int_rate" in df.columns and "loan_grade" in df.columns:
        df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
            lambda x: x.fillna(x.median())
        )

    df = df.dropna()

    # Valida ranges
    df = df[
        (df["person_age"] >= 18) & (df["person_age"] <= 100) & (df["person_income"] > 0)
    ]

    # Metadados
    df["_source"] = "credit_risk_synthetic"
    df["_processed_at"] = pd.Timestamp.now().isoformat()

    df.to_csv(SILVER_DIR / "credit_risk_clean.csv", index=False)
    print(f"  Silver: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    print(f"  Salvo em: data/silver/credit_risk_clean.csv")
    return df


def step2_gold(df_silver):
    """Silver -> Gold: Star Schema e agregacoes."""
    print("\n[2/4] GOLD - Gerando tabelas analiticas...")

    df = df_silver.copy()
    meta_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=meta_cols)

    # Faixa etaria
    df["age_group"] = pd.cut(
        df["person_age"],
        bins=[0, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"],
    ).astype(str)

    # Label legivel do target
    df["loan_status_label"] = df["loan_status"].map(
        {0: "Adimplente", 1: "Inadimplente"}
    )

    # --- fact_loans ---
    fact = df[
        [
            "person_age",
            "person_income",
            "person_home_ownership",
            "person_emp_length",
            "loan_intent",
            "loan_grade",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "loan_status",
            "loan_status_label",
            "cb_person_default_on_file",
            "cb_person_cred_hist_length",
            "age_group",
        ]
    ].copy()
    fact.to_csv(GOLD_DIR / "fact_loans.csv", index=False)
    print(f"  fact_loans: {len(fact):,} registros")

    # --- dim_borrower ---
    dim_b = (
        df[
            [
                "person_age",
                "person_income",
                "person_home_ownership",
                "person_emp_length",
                "cb_person_default_on_file",
                "cb_person_cred_hist_length",
                "age_group",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    dim_b["has_prior_default"] = (dim_b["cb_person_default_on_file"] == "Y").astype(int)
    dim_b.to_csv(GOLD_DIR / "dim_borrower.csv", index=False)
    print(f"  dim_borrower: {len(dim_b):,} registros")

    # --- agg_default_by_grade ---
    agg_grade = (
        df.groupby("loan_grade")
        .agg(
            total_loans=("loan_status", "count"),
            defaults=("loan_status", "sum"),
            default_rate=("loan_status", "mean"),
            avg_int_rate=("loan_int_rate", "mean"),
            avg_loan_amnt=("loan_amnt", "mean"),
            total_volume=("loan_amnt", "sum"),
        )
        .reset_index()
        .rename(columns={"loan_grade": "loan_grade"})
    )
    agg_grade = agg_grade.sort_values("loan_grade")
    agg_grade.to_csv(GOLD_DIR / "agg_default_by_grade.csv", index=False)
    print(f"  agg_default_by_grade: {len(agg_grade)} grades")

    # --- agg_default_by_intent ---
    agg_intent = (
        df.groupby("loan_intent")
        .agg(
            total_loans=("loan_status", "count"),
            defaults=("loan_status", "sum"),
            default_rate=("loan_status", "mean"),
            avg_loan_amnt=("loan_amnt", "mean"),
        )
        .reset_index()
    )
    agg_intent.to_csv(GOLD_DIR / "agg_default_by_intent.csv", index=False)
    print(f"  agg_default_by_intent: {len(agg_intent)} categorias")

    # --- agg_default_by_age ---
    agg_age = (
        df.groupby("age_group")
        .agg(
            total_loans=("loan_status", "count"),
            defaults=("loan_status", "sum"),
            default_rate=("loan_status", "mean"),
            avg_income=("person_income", "mean"),
        )
        .reset_index()
    )
    agg_age.to_csv(GOLD_DIR / "agg_default_by_age.csv", index=False)
    print(f"  agg_default_by_age: {len(agg_age)} faixas")

    print("  Gold completo!")
    return df


def step3_train(df=None):
    """Treina modelos utilizando a suite de Machine Learning de producao."""
    print("\n[3/4] MODELO - Engenharia de Features + Treinamento e Tuning...")
    from src.models.train_model import run_training

    training_results = run_training()
    return training_results


def step4_verify():
    """Verifica se tudo foi gerado corretamente."""
    print("\n[4/4] VERIFICACAO...")
    arquivos_esperados = [
        SILVER_DIR / "credit_risk_clean.csv",
        GOLD_DIR / "fact_loans.csv",
        GOLD_DIR / "agg_default_by_grade.csv",
        GOLD_DIR / "agg_default_by_intent.csv",
        GOLD_DIR / "agg_default_by_age.csv",
        GOLD_DIR / "model_comparison.csv",
        MODELS_DIR / "best_model.pkl",
        MODELS_DIR / "scaler.pkl",
    ]
    ok = True
    for arq in arquivos_esperados:
        status = "OK" if arq.exists() else "FALTANDO"
        tamanho = f"{arq.stat().st_size / 1024:.1f} KB" if arq.exists() else ""
        print(f"  [{status}] {arq.relative_to(ROOT)} {tamanho}")
        if not arq.exists():
            ok = False
    return ok


if __name__ == "__main__":
    print("=" * 55)
    print("PIPELINE DE RISCO DE CREDITO - EXECUCAO COMPLETA")
    print("=" * 55)

    if not BRONZE.exists():
        print(f"\nArquivo Bronze nao encontrado: {BRONZE}")
        print("Executando ingestao oficial via OpenML (ID: 43454)...")
        from src.ingestion.ingest_openml import run_ingestion

        run_ingestion()

    df_silver = step1_silver()
    df_gold = step2_gold(df_silver)
    step3_train()
    ok = step4_verify()

    print("\n" + "=" * 55)
    if ok:
        print("PIPELINE CONCLUIDO!")
        print("\nAgora rode o dashboard:")
        print("  streamlit run src/dashboard/app.py")
    else:
        print("Alguns arquivos nao foram gerados. Veja os erros acima.")
    print("=" * 55)
