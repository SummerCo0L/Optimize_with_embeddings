# Optimizing Resource Allocation using Language Model-Generated Similarity Scores

## Overview

This project is aimed at optimizing the allocation of workers to projects based on various attributes and constraints. 
It consists of two main scripts:

- **Part A**: Analyzes worker and project attributes, generate embeddings with pre-trained language model, calculates scores, and generates an output file.
- **Part B**: Utilizes the output from Part A to optimize the assignment of workers to projects using linear programming techniques.

The project utilizes Python scripts and various utility self-defined functions to process input data, perform calculations, and generate optimized assignments.

## Features

- **Input Processing**: Reads input data from various sheets in Excel file and preprocesses it for analysis.
- **Score Calculation**: Calculates scores based on worker-project attributes and preferences.
- **Optimization**: Utilizes linear programming to optimize the assignment of workers to projects while considering various constraints.
- **Output Generation**: Generates detailed output files containing assignment information and overview tables.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject

2. Install the required dependencies:
   pip install -r requirements.txt
   python setup.py


## Usage
1. Run Part A script to process input data and generate output:
   python part_a.py

2. Run Part B script to optimize worker-project assignments:
   python part_b.py

3. Sample Dataset:
   You can find a sample dataset in the `data/input` directory of this repository. The file is named `sample_dataset.xlsx`.

4. Generated Outputs:
   The outputs of the scripts will be generated in the `data/outputs` directory. After running the scripts, you can find the output files in this directory.
   I've provided some samples for your reference.



