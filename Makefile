.PHONY: install test test-cov lopo-train

install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=emo_mocap --cov-report=term-missing

# --- LOPO cross-validation ---
# Usage:
#   make lopo-train CONFIG=configs/diema12_stgcn.yaml FOLDS=10
FOLDS ?= 10

lopo-train:  ## Train all LOPO folds sequentially
	@for fold in $$(seq 1 $(FOLDS)); do \
		echo "=== Fold $$fold / $(FOLDS) ==="; \
		emo-train --config $(CONFIG) --fold $$fold --num-folds $(FOLDS) --test-after; \
	done
