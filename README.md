# Optimizing Resource Allocation using Language Model-Generated Similarity Scores

## Overview

This project aimed to optimize the allocation of workers to projects based on various attributes and constraints.
It utilizes pre-trained language model to process text data and calculate similarity scores for use in the optimization algorithm.
There are two main scripts:

- **Part A**: Analyzes worker and project attributes, generate embeddings with pre-trained language model, calculates scores, and create an output file.
- **Part B**: Utilizes the output from Part A to optimize the assignment of workers to projects using linear programming techniques.

## Features

- **Input Processing**: Reads input data from various sheets in Excel file and preprocesses it for analysis.
- **Score Calculation**: Calculates scores based on worker-project attributes and preferences.
- **Optimization**: Utilizes linear programming to optimize the assignment of workers to projects while considering various constraints.
- **Output Generation**: Generates detailed output files containing assignment information and overview tables.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SummerCo0L/optimize.git
   cd your_local_path_to/optimize

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Use Transformers offline by downloading the files ahead of time
   ```bash
   python setup.py

## Usage
1. Run Part A script to process input data and generate output:
   ```bash
   python Part_A_Script.py

3. Run Part B script to optimize worker-project assignments:
   ```bash
   python Part_B_Script.py

5. Sample Dataset:
   You can find a sample dataset in the `data/input` directory of this repository. The file is named `Sample_Dataset.xlsx`.

6. Generated Outputs:
   The outputs of the scripts will be generated in the `data/outputs` directory. After running the scripts, you can find the output files in this directory.
   I've provided some samples for your reference.



