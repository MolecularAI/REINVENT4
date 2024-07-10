REINVENT 4 Notebooks
====================

This is a collection of notebooks that we hope will grow over time.  We
strongly encourage our users to contribute their notebooks to the community!

Make sure to run the notebooks in the same environment as you have set up for
REINVENT. You can download [the results](https://www.dropbox.com/scl/fi/s7itk129mdca0s3jgv4qt/R4_notebooks_results.zip?rlkey=ah083im776ut4wel269iihxgz&st=85np6q65&dl=0) from a demo run.

The Python light-script files in this directory need to be converted to a notebook

```shell
jupytext --to ipynb -o Reinvent_demo.ipynb Reinvent_demo.py
```

or run
```shell
./convert_to_notebook.sh
```
to convert all light-script files.

Current notebooks in this series:
- Reinvent\_demo: demos a short RL run focussing on data visualisation and extraction from TensorBoard and the CSV file
- Reinvent\_TLRL.py: shows how a complete workflow with TL model focusing and a stage2 predictive model could work, download the [model](https://www.dropbox.com/scl/fi/zpnqc9at5a5dnkzfdbo6g/model.pt?rlkey=g005yli9364uptd94d60jtg5c&dl=0).
