# Satisfacao de clientes PT-BR
Este repositorio contem alguns modelos treinados para classificar textos de reviews de clientes. Base de treinamento contem 200k registros. 

# run

Para instalar e iniciar o servico execute:

```bash
$ make  run
{...}
python ptsatmodel/run.py --model svm --storage models
Starting SVM model
 * Running on http://0.0.0.0:8000/
```

testando:

```bash
$ curl http://localhost:8000/ -d "text=Péssimo Serviço!!"
"not satisfied"

$ curl http://localhost:8000/ -d "text=Atendentes super atenciosos."
"satisfied"
```

<aside class="notice">
Note: O servico nao esta "production ready"
</aside>


 ### TODO
 - Adicionar Dockerfile
 - Utilizar modelos mais apropriados para classificacao de texto. e.g. DL, fasttext
 - Treinar com toda a base
 
 
