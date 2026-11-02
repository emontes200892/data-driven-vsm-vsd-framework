PYTHON ?= python3

.PHONY: verify reproduce scenarios figures test
verify:
	$(PYTHON) reproduce.py --stage verify
reproduce:
	$(PYTHON) reproduce.py
scenarios:
	$(PYTHON) reproduce.py --stage scenarios
	$(PYTHON) reproduce.py --stage statistics
figures:
	$(PYTHON) reproduce.py --stage figures
test:
	$(PYTHON) -m unittest discover -s tests -v
