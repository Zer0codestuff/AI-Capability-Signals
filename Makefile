PYTHON ?= uv run python

.PHONY: all data validate clean dataset deep-analysis

all:
	$(PYTHON) -m frontier_ai.pipeline

data:
	$(PYTHON) -m frontier_ai.pipeline --skip-plots

validate:
	$(PYTHON) -m unittest discover -s tests

dataset:
	$(PYTHON) -m frontier_ai.dataset_factory

deep-analysis:
	$(PYTHON) -m frontier_ai.deep_analysis

clean:
	rm -rf data/raw data/processed data/dataset data/analysis figures report appendix
