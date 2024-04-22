# Optimizing Resource Allocation with Mixed Integer Linear Programming, using similarity scores generated from language model as decision variable

## Overview

This project aimed to optimize the allocation of workers to projects based on various attributes and constraints.
It utilizes pre-trained language model ("all-MiniLM-L6-v2") to process certain text data to calculate similarity scores for use as a one of the decision variables in the optimization step ("PuLP").
There are two main scripts:

- **Part A**: Analyzes worker and project attributes, generate embeddings with pre-trained language model, calculates scores, and create an output file.
- **Part B**: Utilizes the output from Part A to optimize the assignment of workers to projects using Mixed Integer Linear Programming that involves maximizing an objective function.

## Packages
- Pandas
- NumPy
- Openpyxl
- Pulp
- Tkinter
- Scikit-learn
- Sentence Transformers
- Transformers
- Scipy
- Torch
- 
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

2. Run Part B script to optimize worker-project assignments:
   ```bash
   python Part_B_Script.py

3. Sample Dataset:
   You can find a sample dataset in the `data/input` directory of this repository. The file is named `Sample_Dataset.xlsx`.

4. Generated Outputs:
   The outputs of the scripts will be generated in the `data/outputs` directory. After running the scripts, you can find the output files in this directory.
   I've provided some samples for your reference.


### Customizing Weight Values

Users have the flexibility to customize weight values for different decision variables directly within the Excel input file. 
These weight values can be adjusted according to specific preferences or requirements. 

**Note:** The total sum of weight values for all decision variables should equal 1.

To modify the weight values:

1. Open the input Excel file located at `data/input/Sample_Dataset.xlsx`.
2. Navigate to the relevant sheet ("Weightage").
4. Update the cell values according to your preferences, ensuring that the total sum equals 1.
5. Save the changes to the Excel file.

Part_B script (optimization step) will take into account the updated weight values for generating the final allocation solution.

