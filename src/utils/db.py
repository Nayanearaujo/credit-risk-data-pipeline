"""
Módulo de utilitários: conexão com banco de dados
==================================================
Suporta DuckDB (padrão, analytics local) e PostgreSQL (produção).
Design preparado para migração a Cloud (Azure SQL / AWS RDS).

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from loguru import logger

try:
    from sqlalchemy import create_engine, text
except ImportError:
    create_engine, text = None, None

load_dotenv()


# ---------------------------------------------------------------------------
# DuckDB - banco analítico local (padrão do projeto)
# ---------------------------------------------------------------------------


def get_duckdb_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Retorna uma conexão com o DuckDB local.

    Args:
        read_only: Se True, abre o banco em modo leitura (útil para queries no dashboard).

    Returns:
        Conexão ativa com o DuckDB.
    """
    db_path = os.getenv("DUCKDB_PATH", "./data/credit_risk.duckdb")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(database=db_path, read_only=read_only)
    logger.debug(f"Conexão DuckDB estabelecida: {db_path}")
    return conn


def execute_duckdb_query(query: str, params: dict = None) -> "duckdb.DuckDBPyRelation":
    """
    Executa uma query no DuckDB e retorna o resultado.

    Args:
        query: Query SQL a executar.
        params: Parâmetros opcionais para a query.

    Returns:
        Resultado da query como DataFrame (via .df()).
    """
    with get_duckdb_connection() as conn:
        if params:
            result = conn.execute(query, params)
        else:
            result = conn.execute(query)
        return result.df()


# ---------------------------------------------------------------------------
# PostgreSQL - banco relacional (opcional, para ambiente de produção)
# ---------------------------------------------------------------------------


def get_postgres_engine():
    """
    Cria e retorna uma engine SQLAlchemy para PostgreSQL.

    Variáveis de ambiente necessárias (ver .env.example):
        POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
        POSTGRES_USER, POSTGRES_PASSWORD

    Returns:
        Engine SQLAlchemy conectada ao PostgreSQL.

    Raises:
        EnvironmentError: Se variáveis de ambiente não estiverem configuradas.
        ImportError: Se SQLAlchemy não estiver instalado.
    """
    if create_engine is None:
        raise ImportError(
            "SQLAlchemy não está instalado. Instale com 'pip install sqlalchemy psycopg2-binary'."
        )

    required_vars = [
        "POSTGRES_HOST",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Variáveis de ambiente faltando: {missing}. "
            "Configure o arquivo .env (veja .env.example)."
        )

    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(url, pool_pre_ping=True)
    logger.debug(f"Engine PostgreSQL criada: {host}:{port}/{db}")
    return engine


def test_postgres_connection() -> bool:
    """
    Testa a conexão com o PostgreSQL.

    Returns:
        True se a conexão foi bem-sucedida, False caso contrário.
    """
    try:
        engine = get_postgres_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Conexão PostgreSQL OK")
        return True
    except Exception as e:
        logger.warning(f"⚠️  PostgreSQL não disponível: {e}")
        return False
