"""
Teste de Integração End-to-End do Pipeline de Risco de Crédito
=============================================================
Valida o fluxo completo de dados através das camadas do pipeline:
  Fixture Bruta (Bronze) → Silver → Gold → Feature Engineering → Modelo (Baseline)

AVISO: Utiliza uma fixture sintética de 200 linhas estritamente para
testes automatizados de integração, sem contato com dados de produção.

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.feature_engineering import (
    create_derived_features,
    encode_categoricals,
    select_features,
)
from src.transformation.gold_transform import (
    build_agg_default_by_age,
    build_agg_default_by_grade,
    build_agg_default_by_intent,
    build_dim_borrower,
    build_dim_loan_type,
    build_fact_loans,
)
from src.transformation.silver_transform import (
    add_quality_flags,
    fix_data_types,
    handle_nulls,
    remove_duplicates,
    validate_ranges,
)


@pytest.fixture
def test_bronze_fixture() -> pd.DataFrame:
    """
    Gera uma fixture controlada de 200 registros simulando a camada Bronze.
    Rotulada exclusivamente como fixture de teste de integração.
    """
    np.random.seed(123)
    n = 200
    grades = ["A", "B", "C", "D", "E", "F", "G"]
    intents = [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ]
    homes = ["RENT", "MORTGAGE", "OWN", "OTHER"]

    grade_col = np.random.choice(grades, n)

    df = pd.DataFrame(
        {
            "person_age": np.random.choice(
                [20, 25, 30, 40, 50, 120], n
            ),  # Contém outlier para testar range
            "person_income": np.random.choice(
                [30000, 50000, 80000, 120000, 0], n
            ),  # Contém 0 para testar range
            "person_home_ownership": np.random.choice(homes, n),
            "person_emp_length": np.where(
                np.random.rand(n) < 0.1, np.nan, np.random.randint(0, 20, n)
            ),
            "loan_intent": np.random.choice(intents, n),
            "loan_grade": grade_col,
            "loan_amnt": np.random.choice([2000, 5000, 10000, 15000, 25000], n),
            "loan_int_rate": np.where(
                np.random.rand(n) < 0.1,
                np.nan,
                np.random.uniform(5.0, 25.0, n).round(2),
            ),
            "loan_percent_income": np.random.choice(
                [0.1, 0.2, 0.3, 0.5, 1.5], n
            ),  # Contém > 1 para testar range
            "cb_person_default_on_file": np.random.choice(["N", "Y"], n, p=[0.8, 0.2]),
            "cb_person_cred_hist_length": np.random.randint(1, 15, n),
            "loan_status": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        }
    )

    # Adiciona 5 duplicatas intencionais
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df


class TestEndToEndPipeline:
    """Valida o encadeamento das etapas do pipeline em dados de teste."""

    def test_pipeline_integration_flow(self, test_bronze_fixture, tmp_path):
        # 1. Camada Silver: Limpeza e validação
        df_bronze = test_bronze_fixture.copy()
        initial_len = len(df_bronze)

        df_dedup = remove_duplicates(df_bronze)
        assert len(df_dedup) < initial_len, "Duplicatas deveriam ter sido removidas"

        df_typed = fix_data_types(df_dedup)
        assert pd.api.types.is_numeric_dtype(df_typed["person_income"])
        assert pd.api.types.is_integer_dtype(df_typed["loan_status"])

        df_nulls = handle_nulls(df_typed)
        assert (
            df_nulls["person_emp_length"].isnull().sum() == 0
        ), "Nulos de emp_length deveriam ser preenchidos"
        assert (
            df_nulls["loan_int_rate"].isnull().sum() == 0
        ), "Nulos de loan_int_rate deveriam ser preenchidos"

        df_silver = validate_ranges(df_nulls)
        assert (df_silver["person_age"] <= 100).all(), "Idade > 100 deve ser removida"
        assert (df_silver["person_income"] > 0).all(), "Renda zero deve ser removida"
        assert (
            df_silver["loan_percent_income"] <= 1.0
        ).all(), "Comprometimento > 100% deve ser removido"

        df_silver = add_quality_flags(df_silver)
        assert "_silver_timestamp" in df_silver.columns
        assert "_row_id" in df_silver.columns
        assert len(df_silver) > 0, "Dataset Silver não pode estar vazio"

        # 2. Camada Gold: Modelagem Dimensional
        dim_borrower = build_dim_borrower(df_silver)
        assert "borrower_id" in dim_borrower.columns
        assert len(dim_borrower) == len(df_silver)

        dim_loan_type = build_dim_loan_type(df_silver)
        assert "loan_type_id" in dim_loan_type.columns

        fact_loans = build_fact_loans(df_silver, dim_borrower, dim_loan_type)
        assert "loan_status_label" in fact_loans.columns
        assert set(fact_loans["loan_status_label"].unique()).issubset(
            {"Adimplente", "Inadimplente"}
        )

        agg_grade = build_agg_default_by_grade(df_silver)
        assert "default_rate" in agg_grade.columns

        agg_intent = build_agg_default_by_intent(df_silver)
        assert len(agg_intent) > 0

        agg_age = build_agg_default_by_age(df_silver)
        assert "age_group" in agg_age.columns

        # 3. Engenharia de Features
        df_clean = df_silver.drop(
            columns=[c for c in df_silver.columns if c.startswith("_")], errors="ignore"
        )
        df_derived = create_derived_features(df_clean)
        assert "debt_to_income_ratio" in df_derived.columns
        assert "estimated_total_cost" in df_derived.columns
        assert "has_prior_default" in df_derived.columns

        df_encoded = encode_categoricals(df_derived)
        assert "loan_grade_encoded" in df_encoded.columns

        X, y = select_features(df_encoded)
        assert X.isnull().sum().sum() == 0, "Features não podem conter nulos"
        assert len(X) == len(y)

        # 4. Treinamento de Modelo Rápido (Baseline de Teste: LogisticRegression)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_scaled, y)

        y_proba = model.predict_proba(X_scaled)[:, 1]
        assert len(y_proba) == len(y)
        assert ((y_proba >= 0.0) & (y_proba <= 1.0)).all()

        auc = roc_auc_score(y, y_proba)
        brier = brier_score_loss(y, y_proba)

        assert 0.5 <= auc <= 1.0, f"AUC inválido: {auc}"
        assert 0.0 <= brier <= 1.0, f"Brier score inválido: {brier}"

        # 5. Exportação do relatório comparativo simulado
        comparison_path = tmp_path / "model_comparison.csv"
        df_comparison = pd.DataFrame(
            [
                {
                    "model": "LogisticRegression_IntegrationTest",
                    "test_auc_roc": round(auc, 4),
                    "brier_score": round(brier, 4),
                    "rows_processed": len(X),
                }
            ]
        )
        df_comparison.to_csv(comparison_path, index=False)
        assert comparison_path.exists()
        assert len(pd.read_csv(comparison_path)) == 1
