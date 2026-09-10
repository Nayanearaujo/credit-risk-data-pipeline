"""
Testes — Camada de Ingestão (Bronze)
=====================================
Verifica que os scripts de ingestão funcionam corretamente.

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion.ingest_kaggle import log_bronze_metadata, load_local_dataset


class TestBronzeIngestion:
    """Testes para a camada Bronze."""

    def test_log_bronze_metadata_runs_without_error(self, capsys):
        """Verifica que o logging de metadados não lança exceções."""
        df = pd.DataFrame({
            "person_age": [25, 30, None],
            "loan_amnt": [5000, 10000, 15000],
            "loan_status": [0, 1, 0],
        })
        log_bronze_metadata(df)  # Não deve lançar exceção

    def test_load_local_dataset_raises_when_missing(self, tmp_path):
        """Garante erro informativo quando o arquivo não existe."""
        fake_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
            load_local_dataset(fake_path)

    def test_load_local_dataset_returns_dataframe(self, tmp_path):
        """Verifica que CSV existente é carregado como DataFrame."""
        # Cria CSV temporário
        csv_path = tmp_path / "test.csv"
        df_sample = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df_sample.to_csv(csv_path, index=False)

        result = load_local_dataset(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["col1", "col2"]
