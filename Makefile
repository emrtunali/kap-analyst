.PHONY: install data index eval test lint serve

install:
	pip install -e ".[ml,dev]"

data:
	kap-analyst download
	kap-analyst verify-labels

index:
	kap-analyst index

eval:
	kap-analyst eval

test:
	pytest -q

lint:
	ruff check .

serve:
	kap-analyst serve
