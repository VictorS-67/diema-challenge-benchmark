.PHONY: install test test-cov lpo-train

install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=emo_mocap --cov-report=term-missing

# --- LPO cross-validation ---
# Usage:
#   make lpo-train CONFIG=configs/diema12_stgcn.yaml FOLDS=10
FOLDS ?= 10

lpo-train:  ## Train all LPO folds sequentially
	@for fold in $$(seq 1 $(FOLDS)); do \
		echo "=== Fold $$fold / $(FOLDS) ==="; \
		emo-train --config $(CONFIG) --fold $$fold --num-folds $(FOLDS) --test-after; \
	done
