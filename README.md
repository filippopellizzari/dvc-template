# dvc-template

Project template using DVC and Poetry.

## DVC

DVC, or Data Version Control, is an open-source version control system for machine learning projects. It is designed to handle large files and datasets efficiently, making it easier to manage and version control your machine learning experiments and data.

Key features:

- **Versioning Data**: DVC allows you to version control your datasets separately from your code. Instead of storing large datasets in your version control system (like Git), DVC stores metadata and pointers to the actual data files.

- **Dependency Management**: DVC helps manage dependencies between code, data, and experiments. You can specify dependencies between different stages of your machine learning pipeline, making it easier to reproduce experiments.

- **Reproducibility**: With DVC, you can reproduce any experiment by using the exact code and data that were used initially. This ensures that your experiments are reproducible, a crucial aspect in machine learning research and development.

Quick start:
1. Init DVC
```
dvc init
```
2. Add a remote data repository (e.g. Amazon S3 bucket)
```
dvc remote add -d myremote s3://<bucket>/<key>
```
3. Add data to be versioned
```
dvc add data/Train.csv
```
4. Run experiment pipeline (steps defined in dvc.yaml)
```
dvc exp run
```
5. Push data to remote data repository and code
```
dvc push
```
6. Push code
```
git add .
git commit -m "My Experiment"
git push
```
7. Pro tip: install git hooks, so when you push the code also push the data
```
dvc install

git add .
git commit -m "My Experiment"
git push   (implicit dvc push)
```

Reference [guide](https://dvc.org/doc/user-guide)

## Poetry

Poetry is a dependency management and packaging tool for Python. It simplifies the process of managing project dependencies and packaging by providing a single tool to handle both tasks. Poetry is often used in Python projects to manage project dependencies, create virtual environments, and package Python applications.

Key features:

- **Dependency Management**: Poetry uses a simple and intuitive *pyproject.toml* file to define project metadata, dependencies, and other settings. This file replaces the traditional *requirements.txt* and *setup.py* files.

- **Dependency Resolution**: Poetry automatically resolves and installs dependencies, making it easier to manage complex dependency trees. It uses a lock file (*poetry.lock*) to ensure that dependencies are installed with the exact versions specified.

- **Virtual Environments**: Poetry automatically creates and manages virtual environments for your projects. This helps isolate project dependencies, ensuring that your project's dependencies don't interfere with the global Python environment.

Quick start:
1. Create pyproject and virtual env
```
poetry init
```
2. Activate virtual env
```
poetry shell
```
3. Add package to virtual env
```
poetry add {mypackage}
```
4. Exit virtual env
```
Deactivate
```
Reference [guide](https://python-poetry.org/docs/basic-usage/)