HOST     = localhost
USER     = root
PASSWORD = root
INPUT    = input.csv
STORAGE  = models

.PHONY: build
build:
	pip install -r requirements.txt
	pip install -e .

$(INPUT):
	./bin/dbtocsv.py --host $(HOST) --user $(USER) --password $(PASSWORD)

.PHONY: benchmark
benchmark: build
	python ptsatmodel/benchmarking.py --input $(INPUT) --storage $(STORAGE)

.PHONY: run
run-$(MODEL):
	python ptsatmodel/run.py --model $(MODEL) --storage $(STORAGE)

models/SVM_model.pkl: benchmark
