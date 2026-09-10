# =============================================================================
# Makefile — Atalhos do projeto Credit Risk Data Pipeline
# Autora: Nayane Araújo | github.com/Nayanearaujo
# =============================================================================

.PHONY: help setup run-pipeline run-dashboard test lint clean

help:  ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Instala dependências e configura o ambiente
	@echo "🔧 Configurando ambiente..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "✅ Ambiente pronto! Configure seu .env antes de continuar."

run-pipeline:  ## Executa o pipeline completo Bronze → Silver → Gold
	@echo "🚀 Iniciando pipeline de dados..."
	python src/ingestion/ingest_kaggle.py
	python src/transformation/silver_transform.py
	python src/transformation/gold_transform.py
	@echo "✅ Pipeline concluído!"

run-bronze:  ## Executa apenas a camada Bronze (ingestão)
	python src/ingestion/ingest_kaggle.py
	python src/ingestion/ingest_bcb_api.py

run-silver:  ## Executa apenas a camada Silver (limpeza)
	python src/transformation/silver_transform.py

run-gold:  ## Executa apenas a camada Gold (modelagem dimensional)
	python src/transformation/gold_transform.py

train:  ## Treina o modelo de Machine Learning
	@echo "🤖 Treinando modelo de risco de crédito..."
	python src/models/train_model.py
	python src/models/evaluate_model.py
	@echo "✅ Modelo treinado e avaliado!"

run-dashboard:  ## Inicia o dashboard Streamlit
	@echo "📊 Iniciando dashboard..."
	streamlit run src/dashboard/app.py

test:  ## Executa os testes com cobertura
	@echo "🧪 Executando testes..."
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:  ## Verifica qualidade do código
	@echo "🔍 Verificando código..."
	flake8 src/ tests/ --max-line-length=100
	black --check src/ tests/

format:  ## Formata o código automaticamente
	black src/ tests/

clean:  ## Remove arquivos temporários
	@echo "🧹 Limpando arquivos temporários..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
	@echo "✅ Limpo!"
