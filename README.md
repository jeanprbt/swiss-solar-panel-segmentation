# Solar Panel Segmentation from Swisstopo Aerial Images ☀️
>
> CS-433 Machine Learning - Project 2
>
> Mehdi Zoghlami, Maxime Ducourau, Jean Perbet

## Introduction

This repository contains the code for the second project of the course CS-433 - Machine Learning @ EPFL. This project was run in partnership with the [Swiss Data Science Center](https://www.datascience.ch), and the goal was to develop a machine learning model to detect solar panels from aerial images provided by [Swisstopo](https://www.swisstopo.admin.ch/de).

## Repository structure

The repository is structured as follows:

```foo
├── data/                           
│   ├── images/              # Images
│   ├── labels/              # Ground truth labels
│   │   └── labels.json             
│   ├── roof_coordinates/    # Coordinates of the roofs
│   ├── roof_images/         # Roof-only images
│   ├── tiles/               # Original tiles
│   └── urls/                # Swisstopo URLs
│       └── urls.csv
├── notebooks/
│   └── preprocessing.ipynb  # Notebook for data retrieval & preprocessing
├── src/
│   ├── plotting.py          # Plotting functions
│   └── preprocessing.py     # Preprocessing functions
└── README.md
```

## Run instructions

1. At first, data should be collected and put in the corresponding directories. There are two ways to do this.
    - Run the `preprocessing.ipynb` notebook in the `notebooks/` directory
    - Directly download the data from this [kaggle dataset](https://www.kaggle.com/datasets/jeanperbet/ml-project-2-solar-panels/data?select=roof_images)
