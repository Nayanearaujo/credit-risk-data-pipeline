"""
run_pipeline.py — Roda o pipeline completo de uma vez
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
    print("\n[1/4] SILVER — Limpeza dos dados...")
    df = pd.read_csv(BRONZE)
    print(f"  Bronze: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

    # Remove duplicatas
    antes = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicatas removidas: {antes - len(df)}")

    # Padroniza strings
    str_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
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
        (df["person_age"] >= 18) & (df["person_age"] <= 100) &
        (df["person_income"] > 0)
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
    print("\n[2/4] GOLD — Gerando tabelas analiticas...")

    df = df_silver.copy()
    meta_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=meta_cols)

    # Faixa etaria
    df["age_group"] = pd.cut(
        df["person_age"],
        bins=[0, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"]
    ).astype(str)

    # Label legivel do target
    df["loan_status_label"] = df["loan_status"].map({0: "Adimplente", 1: "Inadimplente"})

    # --- fact_loans ---
    fact = df[[
        "person_age", "person_income", "person_home_ownership",
        "person_emp_length", "loan_intent", "loan_grade",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "loan_status", "loan_status_label",
        "cb_person_default_on_file", "cb_person_cred_hist_length",
        "age_group"
    ]].copy()
    fact.to_csv(GOLD_DIR / "fact_loans.csv", index=False)
    print(f"  fact_loans: {len(fact):,} registros")

    # --- dim_borrower ---
    dim_b = df[["person_age", "person_income", "person_home_ownership",
                "person_emp_length", "cb_person_default_on_file",
                "cb_person_cred_hist_length", "age_group"]].drop_duplicates().copy()
    dim_b["has_prior_default"] = (dim_b["cb_person_default_on_file"] == "Y").astype(int)
    dim_b.to_csv(GOLD_DIR / "dim_borrower.csv", index=False)
    print(f"  dim_borrower: {len(dim_b):,} registros")

    # --- agg_default_by_grade ---
    agg_grade = df.groupby("loan_grade").agg(
        total_loans=("loan_status", "count"),
        defaults=("loan_status", "sum"),
        default_rate=("loan_status", "mean"),
        avg_int_rate=("loan_int_rate", "mean"),
        avg_loan_amnt=("loan_amnt", "mean"),
        total_volume=("loan_amnt", "sum"),
    ).reset_index().rename(columns={"loan_grade": "loan_grade"})
    agg_grade = agg_grade.sort_values("loan_grade")
    agg_grade.to_csv(GOLD_DIR / "agg_default_by_grade.csv", index=False)
    print(f"  agg_default_by_grade: {len(agg_grade)} grades")

    # --- agg_default_by_intent ---
    agg_intent = df.groupby("loan_intent").agg(
        total_loans=("loan_status", "count"),
        defaults=("loan_status", "sum"),
        default_rate=("loan_status", "mean"),
        avg_loan_amnt=("loan_amnt", "mean"),
    ).reset_index()
    agg_intent.to_csv(GOLD_DIR / "agg_default_by_intent.csv", index=False)
    print(f"  agg_default_by_intent: {len(agg_intent)} categorias")

    # --- agg_default_by_age ---
    agg_age = df.groupby("age_group").agg(
        total_loans=("loan_status", "count"),
        defaults=("loan_status", "sum"),
        default_rate=("loan_status", "mean"),
        avg_income=("person_income", "mean"),
    ).reset_index()
    agg_age.to_csv(GOLD_DIR / "agg_default_by_age.csv", index=False)
    print(f"  agg_default_by_age: {len(agg_age)} faixas")

    print("  Gold completo!")
    return df


def step3_train(df):
    """Treina modelo e salva o melhor."""
    print("\n[3/4] MODELO — Feature Engineering + Treinamento...")

    df = df.drop(columns=["loan_status_label", "age_group"], errors="ignore")
    meta_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=meta_cols, errors="ignore")

    # Features derivadas
    df["custo_total"] = df["loan_amnt"] * (1 + df["loan_int_rate"] / 100)
    df["renda_por_emp"] = df["person_income"] / (df["person_emp_length"] + 1)
    df["inadimplencia_previa"] = (df["cb_person_default_on_file"] == "Y").astype(int)

    # Encoding
    df["loan_grade_enc"] = OrdinalEncoder(
        categories=[["A", "B", "C", "D", "E", "F", "G"]]
    ).fit_transform(df[["loan_grade"]])

    intent_dummies = pd.get_dummies(df["loan_intent"], prefix="intent", drop_first=True)
    home_dummies = pd.get_dummies(df["person_home_ownership"], prefix="home", drop_first=True)
    df = pd.concat([df, intent_dummies, home_dummies], axis=1)

    drop_cols = ["loan_grade", "loan_intent", "person_home_ownership",
                 "cb_person_default_on_file", "loan_status"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Treina modelos
    resultados = []
    modelos_treinados = {}

    configs = {
        "Regressao Logistica": LogisticRegression(max_iter=500, random_state=42, C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
    }

    try:
        from xgboost import XGBClassifier
        configs["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42, eval_metric="auc", n_jobs=-1
        )
    except ImportError:
        print("  XGBoost nao instalado, usando RF como melhor modelo")

    for nome, modelo in configs.items():
        modelo.fit(X_train_s, y_train)
        y_pred = modelo.predict(X_test_s)
        y_proba = modelo.predict_proba(X_test_s)[:, 1]
        resultados.append({
            "Modelo": nome,
            "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
        })
        modelos_treinados[nome] = modelo
        print(f"  {nome}: AUC={resultados[-1]['auc_roc']:.4f} | F1={resultados[-1]['f1_score']:.4f}")

    # Salva comparacao
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(GOLD_DIR / "model_comparison.csv", index=False)

    # Melhor modelo por AUC
    melhor_nome = df_res.loc[df_res["auc_roc"].idxmax(), "Modelo"]
    melhor = modelos_treinados[melhor_nome]
    print(f"\n  Melhor modelo: {melhor_nome} (AUC={df_res['auc_roc'].max():.4f})")

    # Salva
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(melhor, f)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    pd.Series(feature_cols).to_csv(MODELS_DIR / "feature_names.csv", index=False, header=False)

    print(f"  Modelo salvo: src/models/best_model.pkl")
    print(f"  Scaler salvo: src/models/scaler.pkl")
    return melhor, scaler, feature_cols, X_test_s, y_test


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
    print("PIPELINE DE RISCO DE CREDITO — EXECUCAO COMPLETA")
    print("=" * 55)

    if not BRONZE.exists():
        print(f"\nERRO: Arquivo Bronze nao encontrado: {BRONZE}")
        print("Execute primeiro: python gera_dados.py")
        exit(1)

    df_silver = step1_silver()
    df_gold = step2_gold(df_silver)
    step3_train(df_gold)
    ok = step4_verify()

    print("\n" + "=" * 55)
    if ok:
        print("PIPELINE CONCLUIDO!")
        print("\nAgora rode o dashboard:")
        print("  streamlit run src/dashboard/app.py")
    else:
        print("Alguns arquivos nao foram gerados. Veja os erros acima.")
    print("=" * 55)
