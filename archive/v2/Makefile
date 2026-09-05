PYTHON ?= uv run python

.PHONY: all analyse report validate clean

all:
	$(PYTHON) -m aicap.cli

analyse:
	$(PYTHON) -m aicap.cli --from-interim

report:
	$(PYTHON) -m aicap.cli --from-interim

validate:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf data/interim data/analysis figures report/frontier_signals.md report/frontier_signals.html report/run_manifest.json
