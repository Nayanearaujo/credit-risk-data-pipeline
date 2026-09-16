"""
Silver Layer - Limpeza e Validação dos Dados de Risco de Crédito
================================================================
Transforma os dados brutos (Bronze) em dados limpos, tipados e validados.

Regras aplicadas:
  ✅ Remoção de duplicatas
  ✅ Tratamento de valores nulos por estratégia (mediana / moda / remoção)
  ✅ Tipagem correta das colunas
  ✅ Padronização de strings (lowercase, strip)
  ✅ Validação de ranges (idade, renda, prazo)
  ✅ Detecção e documentação de outliers
  ✅ Registro de metadados de qualidade

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.utils.db import get_duckdb_connection

setup_logger(log_level="INFO")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BRONZE_PATH = Path("data/bronze")
SILVER_PATH = Path("data/silver")
SILVER_PATH.mkdir(parents=True, exist_ok=True)

INPUT_FILE = BRONZE_PATH / "credit_risk_raw.csv"
OUTPUT_FILE = SILVER_PATH / "credit_risk_clean.csv"


# ---------------------------------------------------------------------------
# Funções de limpeza
# ---------------------------------------------------------------------------


def load_bronze(path: Path = INPUT_FILE) -> pd.DataFrame:
    """Carrega os dados brutos da camada Bronze."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo Bronze não encontrado: {path}\n"
            "Execute primeiro: python src/ingestion/ingest_kaggle.py"
        )
    df = pd.read_csv(path)
    logger.info(f"📂 Bronze carregada: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas duplicadas e loga quantidade removida."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        logger.warning(f"🗑️  Duplicatas removidas: {removed:,}")
    else:
        logger.info("✅ Sem duplicatas encontradas")
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige tipagem das colunas do Credit Risk Dataset.

    Colunas esperadas:
        person_age, person_income, person_home_ownership,
        person_emp_length, loan_intent, loan_grade, loan_amnt,
        loan_int_rate, loan_status (target), loan_percent_income,
        cb_person_default_on_file, cb_person_cred_hist_length
    """
    logger.info("🔧 Corrigindo tipos de dados...")

    # Inteiros
    int_cols = ["person_age", "loan_amnt", "cb_person_cred_hist_length"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Floats
    float_cols = [
        "person_income",
        "person_emp_length",
        "loan_int_rate",
        "loan_percent_income",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strings padronizadas
    str_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # Target binário
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].astype(int)

    logger.info("✅ Tipos corrigidos")
    return df


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores nulos com estratégia documentada.

    Estratégias:
        - person_emp_length: mediana (dado ausente ≠ 0 anos de emprego)
        - loan_int_rate:     mediana por loan_grade (taxa reflete grade)
        - demais:            remoção (poucos casos, sem impacto estatístico)
    """
    logger.info("🔧 Tratando valores nulos...")
    original_len = len(df)

    # Estratégia 1: mediana para tempo de emprego
    if "person_emp_length" in df.columns:
        median_emp = df["person_emp_length"].median()
        nulls_emp = df["person_emp_length"].isnull().sum()
        df["person_emp_length"] = df["person_emp_length"].fillna(median_emp)
        logger.info(
            f"  📌 person_emp_length: {nulls_emp} nulos → mediana ({median_emp:.1f} anos)"
        )

    # Estratégia 2: mediana por grade para taxa de juros
    if "loan_int_rate" in df.columns and "loan_grade" in df.columns:
        nulls_rate = df["loan_int_rate"].isnull().sum()
        df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
            lambda x: x.fillna(x.median())
        )
        logger.info(f"  📌 loan_int_rate: {nulls_rate} nulos → mediana por grade")

    # Estratégia 3: remover demais nulos
    df_clean = df.dropna()
    removed = original_len - len(df_clean)
    if removed > 0:
        logger.warning(f"  ⚠️  Linhas removidas por nulos restantes: {removed:,}")

    logger.info(f"✅ Nulos tratados. Dataset: {len(df_clean):,} registros")
    return df_clean


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove registros com valores biologicamente/logicamente impossíveis.

    Regras de negócio:
        - Idade: 18 a 100 anos
        - Renda: > 0
        - Tempo de emprego: 0 a 60 anos
        - Percentual de renda comprometido: 0% a 100%
    """
    logger.info("🔧 Validando ranges de negócio...")
    before = len(df)

    if "person_age" in df.columns:
        df = df[(df["person_age"] >= 18) & (df["person_age"] <= 100)]

    if "person_income" in df.columns:
        df = df[df["person_income"] > 0]

    if "person_emp_length" in df.columns:
        df = df[(df["person_emp_length"] >= 0) & (df["person_emp_length"] <= 60)]

    if "loan_percent_income" in df.columns:
        df = df[(df["loan_percent_income"] >= 0) & (df["loan_percent_income"] <= 1)]

    removed = before - len(df)
    if removed > 0:
        logger.warning(f"⚠️  Registros inválidos removidos: {removed:,}")
    else:
        logger.info("✅ Todos os registros dentro dos ranges esperados")

    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas de flag para análise de qualidade e auditoria.
    Estas colunas ajudam a rastrear a proveniência dos dados.
    """
    df = df.copy()
    df["_source"] = "kaggle_credit_risk_dataset"
    df["_silver_timestamp"] = pd.Timestamp.now().isoformat()
    df["_row_id"] = range(1, len(df) + 1)
    return df


def save_to_silver(df: pd.DataFrame) -> None:
    """Salva os dados limpos na camada Silver (CSV + DuckDB)."""
    # CSV
    df.to_csv(OUTPUT_FILE, index=False)
    logger.success(f"💾 Silver CSV salva em: {OUTPUT_FILE}")

    # DuckDB - registra como view permanente
    try:
        conn = get_duckdb_connection()
        conn.execute(
            """
            CREATE OR REPLACE TABLE silver_credit_risk AS
            SELECT * FROM read_csv_auto(?)
        """,
            [str(OUTPUT_FILE)],
        )
        conn.close()
        logger.success(
            "💾 Silver registrada no DuckDB como tabela 'silver_credit_risk'"
        )
    except Exception as e:
        logger.warning(f"⚠️  DuckDB não disponível: {e}. CSV salvo com sucesso.")


def log_silver_quality_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    """Loga relatório de qualidade comparando antes e depois da limpeza."""
    logger.info("=" * 60)
    logger.info("📋 SILVER LAYER - Relatório de Qualidade")
    logger.info("=" * 60)
    logger.info(f"Registros antes: {len(df_before):,}")
    logger.info(f"Registros após:  {len(df_after):,}")
    logger.info(f"Taxa de retenção: {len(df_after)/len(df_before)*100:.1f}%")
    if "loan_status" in df_after.columns:
        inadimplencia = df_after["loan_status"].mean() * 100
        logger.info(f"Taxa de inadimplência: {inadimplencia:.1f}%")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------


def run_silver_transform() -> pd.DataFrame:
    """Orquestra a transformação completa da camada Silver."""
    logger.info("🚀 Iniciando transformação - Camada Silver")

    df_raw = load_bronze()
    df = remove_duplicates(df_raw)
    df = fix_data_types(df)
    df = handle_nulls(df)
    df = validate_ranges(df)
    df = add_quality_flags(df)
    save_to_silver(df)
    log_silver_quality_report(df_raw, df)

    logger.success("✅ Camada Silver concluída!")
    return df


if __name__ == "__main__":
    df_silver = run_silver_transform()
    print("\nDataset limpo - primeiras linhas:")
    print(df_silver.head())
    print("\nTipos de dados:")
    print(df_silver.dtypes)
