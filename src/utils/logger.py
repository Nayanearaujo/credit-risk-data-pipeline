"""
Módulo de utilitários: configuração de logging padronizado
==========================================================
Usa Loguru para logs ricos com nível, timestamp e rastreamento.

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Configura o logger padronizado do projeto.

    Args:
        log_level: Nível mínimo de log (DEBUG, INFO, WARNING, ERROR).
        log_file:  Caminho para arquivo de log (opcional).
                   Se None, loga apenas no console.
    """
    logger.remove()  # Remove handler padrão

    # Console - colorido e legível
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Arquivo - rotação diária, máximo 10 dias de histórico
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="1 day",
            retention="10 days",
            compression="zip",
        )
        logger.info(f"Logs sendo gravados em: {log_file}")


# Configuração padrão ao importar o módulo
setup_logger()
