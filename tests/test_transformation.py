"""
Testes unitários - Lógica de Transformação (Silver)
====================================================
Testa as funções de produção de src.transformation.silver_transform.

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transformation.silver_transform import (
    remove_duplicates,
    fix_data_types,
    handle_nulls,
    validate_ranges,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Dataset de exemplo completo para os testes."""
    return pd.DataFrame(
        {
            "person_age": [25, 30, 150, 25, 30],
            "person_income": [50_000.0, 80_000.0, 30_000.0, 50_000.0, 80_000.0],
            "person_home_ownership": ["rent", "own", "mortgage", "rent", "own"],
            "person_emp_length": [3.0, None, 5.0, 3.0, None],
            "loan_intent": [
                "personal",
                "education",
                "medical",
                "personal",
                "education",
            ],
            "loan_grade": ["A", "B", "C", "A", "B"],
            "loan_amnt": [5_000, 10_000, 15_000, 5_000, 10_000],
            "loan_int_rate": [7.5, None, 12.0, 7.5, None],
            "loan_status": [0, 1, 0, 0, 1],
            "loan_percent_income": [0.10, 0.12, 0.50, 0.10, 0.12],
            "cb_person_default_on_file": ["N", "Y", "N", "N", "Y"],
            "cb_person_cred_hist_length": [3, 7, 5, 3, 7],
        }
    )


# ---------------------------------------------------------------------------
# Testes: RemoveDuplicates
# ---------------------------------------------------------------------------


class TestRemoveDuplicates:
    def test_remove_duplicatas_exatas(self, sample_df):
        """Linhas identicas devem ser removidas."""
        resultado = remove_duplicates(sample_df)
        assert len(resultado) < len(sample_df)
        assert len(resultado) == len(sample_df.drop_duplicates())

    def test_dataset_sem_duplicatas_inalterado(self):
        """Dataset sem duplicatas nao deve perder registros."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert len(remove_duplicates(df)) == 3

    def test_todas_duplicatas_removidas(self):
        """Duplicatas multiplas: apenas uma deve permanecer."""
        df = pd.DataFrame({"col": ["a", "a", "a", "b"]})
        assert len(remove_duplicates(df)) == 2


# ---------------------------------------------------------------------------
# Testes: FixDataTypes
# ---------------------------------------------------------------------------


class TestFixDataTypes:
    def test_strings_em_maiusculo(self, sample_df):
        """Colunas de string devem ser convertidas para maiusculo."""
        df_unico = sample_df.drop_duplicates().copy()
        resultado = fix_data_types(df_unico)
        assert resultado["person_home_ownership"].str.isupper().all()
        assert resultado["loan_grade"].str.isupper().all()

    def test_renda_como_float(self, sample_df):
        """person_income deve ser tipo numerico."""
        resultado = fix_data_types(sample_df.drop_duplicates().copy())
        assert pd.api.types.is_numeric_dtype(resultado["person_income"])

    def test_target_como_inteiro(self, sample_df):
        """loan_status deve ser inteiro (0 ou 1)."""
        resultado = fix_data_types(sample_df.drop_duplicates().copy())
        assert pd.api.types.is_integer_dtype(resultado["loan_status"])


# ---------------------------------------------------------------------------
# Testes: ValidateRanges
# ---------------------------------------------------------------------------


class TestValidateRanges:
    def test_remove_idade_invalida(self, sample_df):
        """Idades acima de 100 ou abaixo de 18 devem ser removidas."""
        resultado = validate_ranges(sample_df.drop_duplicates().copy())
        assert (resultado["person_age"] <= 100).all()
        assert (resultado["person_age"] >= 18).all()

    def test_remove_renda_zero(self):
        """Renda zero deve ser removida."""
        df = pd.DataFrame(
            {
                "person_age": [25, 30],
                "person_income": [0, 50_000],
                "person_emp_length": [3.0, 5.0],
                "loan_percent_income": [0.1, 0.2],
            }
        )
        resultado = validate_ranges(df)
        assert len(resultado) == 1
        assert resultado["person_income"].iloc[0] == 50_000

    def test_mantém_registros_validos(self):
        """Registros dentro dos ranges devem ser preservados."""
        df = pd.DataFrame(
            {
                "person_age": [20, 35, 50],
                "person_income": [30_000, 60_000, 90_000],
            }
        )
        resultado = validate_ranges(df)
        assert len(resultado) == 3

    def test_comprometimento_acima_de_um_removido(self):
        """Comprometimento de renda acima de 100% deve ser removido."""
        df = pd.DataFrame(
            {
                "person_emp_length": [3.0, 3.0],
                "loan_percent_income": [0.5, 1.5],
            }
        )
        resultado = validate_ranges(df)
        assert len(resultado) == 1


# ---------------------------------------------------------------------------
# Testes: HandleNulls
# ---------------------------------------------------------------------------


class TestHandleNulls:
    def test_sem_nulos_apos_tratamento(self, sample_df):
        """Nenhum nulo deve restar apos o tratamento."""
        df = fix_data_types(sample_df.drop_duplicates().copy())
        resultado = handle_nulls(df)
        assert resultado.isnull().sum().sum() == 0

    def test_mediana_preenche_emp_length(self):
        """Nulos em person_emp_length devem ser preenchidos com mediana."""
        df = pd.DataFrame(
            {
                "person_emp_length": [2.0, 4.0, None, 6.0],
                "loan_grade": ["A", "A", "A", "A"],
                "loan_int_rate": [7.0, 8.0, 9.0, 10.0],
            }
        )
        resultado = handle_nulls(df)
        assert resultado["person_emp_length"].isnull().sum() == 0

    def test_tamanho_reduzido_com_nulos(self, sample_df):
        """Dataset apos tratamento deve ter menos ou igual registros."""
        df_unico = sample_df.drop_duplicates().copy()
        df_typed = fix_data_types(df_unico)
        resultado = handle_nulls(df_typed)
        assert len(resultado) <= len(df_unico)
