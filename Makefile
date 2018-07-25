HOST     = localhost
USER     = root
PASSWORD = root
INPUT    = input.csv
STORAGE  = models
MODEL    = svm

.PHONY: build
build:  models/SVM_model.pkl models/MultinomialNaiveBayes_model.pkl
	pip install -r requirements.txt
	pip install -e .

$(INPUT):
	./bin/dbtocsv.py --host $(HOST) --user $(USER) --password $(PASSWORD)

.PHONY: train
train:
	python ptsatmodel/benchmarking.py --input $(INPUT) --storage $(STORAGE)

.PHONY: run
run: build
	python ptsatmodel/run.py --model $(MODEL) --storage $(STORAGE)

models/SVM_model.pkl:
	train

models/MultinomialNaiveBayes_model.pkl:
	train

