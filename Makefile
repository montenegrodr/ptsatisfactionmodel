HOST     = localhost
USER     = root
PASSWORD = root
INPUT    = input.csv
STORAGE  = models

$(INPUT):
	./bin/dbtocsv.py --host $(HOST) --user $(USER) --password $(PASSWORD)

.PHONY: run
run:
	python clf_benchmarking.py --input $(INPUT) --storage $(STORAGE)
