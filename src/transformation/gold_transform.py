"""
Gold Layer - Modelagem Dimensional e Agregações de Negócio
==========================================================
Transforma os dados limpos (Silver) em dados prontos para:
  - Consumo em BI (Power BI, Streamlit)
  - Alimentação de modelos de Machine Learning
  - Respostas a perguntas de negócio

Modelagem: Star Schema simplificado
  - Fato: fact_loans (empréstimos)
  - Dimensões: dim_borrower, dim_loan_type, dim_risk_profile

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.utils.db import get_duckdb_connection

setup_logger(log_level="INFO")

SILVER_PATH = Path("data/silver")
GOLD_PATH = Path("data/gold")
GOLD_PATH.mkdir(parents=True, exist_ok=True)

INPUT_FILE = SILVER_PATH / "credit_risk_clean.csv"


# ---------------------------------------------------------------------------
# Dimensões
# ---------------------------------------------------------------------------


def build_dim_borrower(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dimensão Tomador de Empréstimo (Borrower).

    Agrupa características demográficas e financeiras do solicitante.
    Em um modelo real, cada linha teria um ID único de cliente.
    """
    logger.info("🏗️  Construindo dim_borrower...")

    dim = df[
        [
            "person_age",
            "person_income",
            "person_home_ownership",
            "person_emp_length",
            "cb_person_default_on_file",
            "cb_person_cred_hist_length",
        ]
    ].copy()

    # Faixas etárias - segmentação para análise de negócio
    dim["age_group"] = pd.cut(
        dim["person_age"],
        bins=[17, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"],
    )

    # Faixas de renda
    dim["income_band"] = pd.cut(
        dim["person_income"],
        bins=[0, 30_000, 60_000, 100_000, float("inf")],
        labels=["Baixa", "Média", "Alta", "Muito Alta"],
    )

    # Histórico de inadimplência prévia
    dim["has_prior_default"] = (dim["cb_person_default_on_file"] == "Y").astype(int)

    dim["borrower_id"] = range(1, len(dim) + 1)
    dim = dim.reset_index(drop=True)

    logger.info(f"✅ dim_borrower: {len(dim):,} registros")
    return dim


def build_dim_loan_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dimensão Tipo de Empréstimo.

    Categoriza o propósito e a classificação de risco do empréstimo.
    """
    logger.info("🏗️  Construindo dim_loan_type...")

    dim = df[["loan_intent", "loan_grade"]].drop_duplicates().copy()
    dim["loan_type_id"] = range(1, len(dim) + 1)

    # Mapeamento de grade para risco
    grade_risk_map = {
        "A": "Muito Baixo",
        "B": "Baixo",
        "C": "Médio",
        "D": "Alto",
        "E": "Muito Alto",
        "F": "Crítico",
        "G": "Crítico",
    }
    dim["risk_level"] = dim["loan_grade"].map(grade_risk_map)

    logger.info(f"✅ dim_loan_type: {len(dim):,} categorias únicas")
    return dim


def build_fact_loans(
    df: pd.DataFrame,
    dim_borrower: pd.DataFrame,
    dim_loan_type: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tabela Fato: Empréstimos.

    Contém as métricas quantitativas de cada operação de crédito,
    com chaves estrangeiras para as dimensões.
    """
    logger.info("🏗️  Construindo fact_loans...")

    # Merge com dimensão tipo de empréstimo
    fact = df.merge(
        dim_loan_type[["loan_intent", "loan_grade", "loan_type_id"]],
        on=["loan_intent", "loan_grade"],
        how="left",
    )

    # Adiciona chave do tomador (posicional neste exemplo)
    fact["borrower_id"] = dim_borrower["borrower_id"].values

    # Seleciona apenas as colunas da fato
    fact = fact[
        [
            "borrower_id",
            "loan_type_id",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "loan_status",
        ]
    ].copy()

    # Métricas derivadas
    fact["annual_interest_cost"] = fact["loan_amnt"] * fact["loan_int_rate"] / 100
    fact["loan_status_label"] = fact["loan_status"].map(
        {0: "Adimplente", 1: "Inadimplente"}
    )

    logger.info(f"✅ fact_loans: {len(fact):,} registros")
    return fact


# ---------------------------------------------------------------------------
# Agregações de negócio
# ---------------------------------------------------------------------------


def build_agg_default_by_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregação: Taxa de inadimplência por grade de risco.
    Responde: 'Qual grade tem maior risco?'
    """
    agg = (
        df.groupby("loan_grade")
        .agg(
            total_loans=("loan_status", "count"),
            defaults=("loan_status", "sum"),
            total_amount=("loan_amnt", "sum"),
            avg_interest_rate=("loan_int_rate", "mean"),
        )
        .reset_index()
    )
    agg["default_rate"] = agg["defaults"] / agg["total_loans"]
    agg = agg.sort_values("default_rate", ascending=False)
    return agg


def build_agg_default_by_intent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregação: Taxa de inadimplência por finalidade do empréstimo.
    Responde: 'Qual propósito de empréstimo é mais arriscado?'
    """
    agg = (
        df.groupby("loan_intent")
        .agg(
            total_loans=("loan_status", "count"),
            defaults=("loan_status", "sum"),
            avg_amount=("loan_amnt", "mean"),
        )
        .reset_index()
    )
    agg["default_rate"] = agg["defaults"] / agg["total_loans"]
    agg = agg.sort_values("default_rate", ascending=False)
    return agg


def build_agg_default_by_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregação: Taxa de inadimplência por faixa etária.
    Responde: 'Qual faixa etária tem maior risco de inadimplência?'
    """
    df = df.copy()
    df["age_group"] = pd.cut(
        df["person_age"],
        bins=[17, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"],
    )
    agg = (
        df.groupby("age_group", observed=True)
        .agg(
            total_loans=("loan_status", "count"),
            defaults=("loan_status", "sum"),
            avg_income=("person_income", "mean"),
        )
        .reset_index()
    )
    agg["default_rate"] = agg["defaults"] / agg["total_loans"]
    return agg


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------


def save_gold_tables(tables: dict) -> None:
    """
    Salva todas as tabelas Gold em CSV e DuckDB.

    Args:
        tables: Dicionário {nome_tabela: DataFrame}
    """
    conn = None
    try:
        conn = get_duckdb_connection()
    except Exception as e:
        logger.warning(f"⚠️  DuckDB não disponível: {e}")

    for name, df in tables.items():
        # CSV
        path = GOLD_PATH / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.success(f"💾 {name} → {path}")

        # DuckDB
        if conn:
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE {name} AS
                SELECT * FROM read_csv_auto('{path}')
            """
            )
            logger.success(f"💾 {name} registrada no DuckDB")

    if conn:
        conn.close()


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------


def run_gold_transform() -> dict:
    """Orquestra a transformação completa da camada Gold."""
    logger.info("🚀 Iniciando transformação - Camada Gold")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Silver não encontrada: {INPUT_FILE}\n"
            "Execute primeiro: python src/transformation/silver_transform.py"
        )

    df = pd.read_csv(INPUT_FILE)
    logger.info(f"📂 Silver carregada: {df.shape[0]:,} registros")

    # Dimensões
    dim_borrower = build_dim_borrower(df)
    dim_loan_type = build_dim_loan_type(df)

    # Fato
    fact_loans = build_fact_loans(df, dim_borrower, dim_loan_type)

    # Agregações
    agg_by_grade = build_agg_default_by_grade(df)
    agg_by_intent = build_agg_default_by_intent(df)
    agg_by_age = build_agg_default_by_age(df)

    tables = {
        "dim_borrower": dim_borrower,
        "dim_loan_type": dim_loan_type,
        "fact_loans": fact_loans,
        "agg_default_by_grade": agg_by_grade,
        "agg_default_by_intent": agg_by_intent,
        "agg_default_by_age": agg_by_age,
    }

    save_gold_tables(tables)

    logger.success("✅ Camada Gold concluída!")
    logger.info("📊 Tabelas disponíveis: " + ", ".join(tables.keys()))
    return tables


if __name__ == "__main__":
    tables = run_gold_transform()
    print("\n📊 Amostra - Taxa de inadimplência por grade:")
    print(tables["agg_default_by_grade"])
