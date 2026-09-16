"""
Compatibilidade retroativa: redireciona para ingest_openml.py
=============================================================
A ingestão primária foi migrada para a OpenML (dataset ID 43454).
Este arquivo mantém as funções disponíveis para compatibilidade.
"""

from src.ingestion.ingest_openml import (
    download_openml,
    download_fallback as download_dataset,
    load_local_dataset,
    log_bronze_metadata,
    run_ingestion,
)

if __name__ == "__main__":
    run_ingestion()
