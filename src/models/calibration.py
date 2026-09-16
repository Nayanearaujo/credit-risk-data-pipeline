"""
Machine Learning - Calibração de Probabilidades
===============================================
Avalia e calibra as probabilidades previstas para concessão de crédito.
Em risco de crédito, 30% de probabilidade prevista deve corresponder a
~30% de inadimplência real observada (empírica).

Métricas e análises:
  - Reliability Curve (Curva de Calibração)
  - Brier Score (perda quadrática das probabilidades)
  - Calibração pós-treino via CalibratedClassifierCV

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

# Adiciona o root do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger

setup_logger(log_level="INFO")

DOCS_PATH = Path("docs")
DOCS_PATH.mkdir(parents=True, exist_ok=True)


def evaluate_calibration(
    model,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str = "XGBoost",
    save_plot: bool = True,
) -> dict:
    """
    Avalia a calibração de um modelo treinado e gera a curva de confiabilidade.

    Args:
        model: Modelo treinado (predict_proba).
        X_test: Features de teste.
        y_test: Labels reais.
        model_name: Nome do modelo.
        save_plot: Se True, salva em docs/calibration_curve.png.

    Returns:
        Dicionário com brier_score, brier_calibrated, e detalhes.
    """
    logger.info(f"📊 Avaliando calibração para {model_name}...")

    y_proba_raw = model.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, y_proba_raw)
    prob_true_raw, prob_pred_raw = calibration_curve(
        y_test, y_proba_raw, n_bins=10, strategy="uniform"
    )

    # Calibração pós-treino (sigmoid / Platt scaling)
    calibrated_clf = CalibratedClassifierCV(
        estimator=model, method="sigmoid", cv="prefit"
    )
    calibrated_clf.fit(X_test, y_test)
    y_proba_cal = calibrated_clf.predict_proba(X_test)[:, 1]
    brier_cal = brier_score_loss(y_test, y_proba_cal)
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_test, y_proba_cal, n_bins=10, strategy="uniform"
    )

    logger.info(f"  📈 Brier Score Original:   {brier_raw:.4f}")
    logger.info(f"  📈 Brier Score Calibrado:  {brier_cal:.4f}")

    if save_plot:
        plot_path = DOCS_PATH / "calibration_curve.png"
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot([0, 1], [0, 1], "k--", label="Perfeitamente Calibrado", lw=1.5)
        ax.plot(
            prob_pred_raw,
            prob_true_raw,
            "s-",
            color="#ef4444",
            label=f"{model_name} Original (Brier={brier_raw:.4f})",
            lw=2,
        )
        ax.plot(
            prob_pred_cal,
            prob_true_cal,
            "o-",
            color="#10b981",
            label=f"{model_name} Calibrado (Brier={brier_cal:.4f})",
            lw=2,
        )

        ax.set_ylabel("Fração Real de Inadimplência", fontsize=12)
        ax.set_xlabel("Probabilidade Média Prevista", fontsize=12)
        ax.set_title(
            f"Curva de Calibração (Reliability Curve) - {model_name}",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.success(f"💾 Gráfico de calibração salvo em: {plot_path}")

    return {
        "model": model_name,
        "brier_score_raw": round(brier_raw, 4),
        "brier_score_calibrated": round(brier_cal, 4),
        "calibrated_model": calibrated_clf,
    }
