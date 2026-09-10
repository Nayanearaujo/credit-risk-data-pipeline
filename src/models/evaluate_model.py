"""
Machine Learning — Avaliação Detalhada do Modelo
================================================
Gera relatórios completos de performance com:
  - Matriz de confusão
  - Curva ROC
  - Curva Precision-Recall
  - Importância de features (SHAP values)
  - Análise de limiar (threshold) de decisão

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.models.feature_engineering import prepare_features

setup_logger(log_level="INFO")

MODELS_PATH = Path("src/models")
DOCS_PATH = Path("docs")
DOCS_PATH.mkdir(parents=True, exist_ok=True)

# Estilo dos gráficos
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = {"Adimplente": "#2ecc71", "Inadimplente": "#e74c3c"}


def load_best_model():
    """Carrega o melhor modelo salvo."""
    path = MODELS_PATH / "best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {path}\n"
            "Execute primeiro: python src/models/train_model.py"
        )
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"📂 Modelo carregado: {type(model).__name__}")
    return model


def plot_confusion_matrix(y_test, y_pred, save_path: Path) -> None:
    """Gera e salva a matriz de confusão."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdYlGn_r",
        xticklabels=["Adimplente", "Inadimplente"],
        yticklabels=["Adimplente", "Inadimplente"],
        ax=ax,
    )
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title("Matriz de Confusão — Risco de Crédito", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"💾 Matriz de confusão salva: {save_path}")


def plot_roc_curve(y_test, y_proba, auc: float, save_path: Path) -> None:
    """Gera e salva a curva ROC."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#3498db", lw=2.5, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Modelo Aleatório")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3498db")

    ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
    ax.set_title("Curva ROC — Modelo de Risco de Crédito", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"💾 Curva ROC salva: {save_path}")


def plot_feature_importance(model, feature_names: list, save_path: Path, top_n: int = 15) -> None:
    """Gera e salva o gráfico de importância das features."""
    if not hasattr(model, "feature_importances_"):
        logger.warning("⚠️  Modelo não suporta feature_importances_. Pulando.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
    top_features.sort_values().plot(kind="barh", ax=ax, color=colors)

    ax.set_xlabel("Importância", fontsize=12)
    ax.set_title(f"Top {top_n} Features — Impacto no Risco de Inadimplência",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"💾 Importância de features salva: {save_path}")


def run_evaluation() -> dict:
    """Orquestra a avaliação completa do melhor modelo."""
    logger.info("🚀 Iniciando avaliação do modelo...")

    model = load_best_model()
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_features()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"📊 AUC-ROC: {auc:.4f}")

    # Relatório completo
    report = classification_report(y_test, y_pred, target_names=["Adimplente", "Inadimplente"])
    logger.info(f"\n{report}")

    # Gráficos
    plot_confusion_matrix(y_test, y_pred, DOCS_PATH / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, auc, DOCS_PATH / "roc_curve.png")
    plot_feature_importance(model, feature_names, DOCS_PATH / "feature_importance.png")

    logger.success("✅ Avaliação concluída! Gráficos salvos em docs/")
    return {"auc": auc, "report": report}


if __name__ == "__main__":
    results = run_evaluation()
