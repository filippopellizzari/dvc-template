# dvc-example

Project template using DVC and Poetry.

## DVC

DVC, or Data Version Control, is an open-source version control system for machine learning projects. It is designed to handle large files and datasets efficiently, making it easier to manage and version control your machine learning experiments and data.

Key features:

- **Versioning Data**: DVC allows you to version control your datasets separately from your code. Instead of storing large datasets in your version control system (like Git), DVC stores metadata and pointers to the actual data files.

- **Dependency Management**: DVC helps manage dependencies between code, data, and experiments. You can specify dependencies between different stages of your machine learning pipeline, making it easier to reproduce experiments.

- **Reproducibility**: With DVC, you can reproduce any experiment by using the exact code and data that were used initially. This ensures that your experiments are reproducible, a crucial aspect in machine learning research and development.

