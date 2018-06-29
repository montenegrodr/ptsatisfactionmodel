HOST     = localhost
USER     = root
PASSWORD = root
INPUT    = input.csv

$(INPUT):
	./bin/dbtocsv.py --host $(HOST) --user $(USER) --password $(PASSWORD)
