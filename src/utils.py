import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F



# =============================================================================
# Script A Functions
# =============================================================================

# Function: Makes a subfolder safely
def makeSubfolder(subfoldername):
    pathlib.Path(subfoldername).mkdir(parents=True, exist_ok=True)
        
	
# Function: Convert excel page to df
def createDf(data):
    dataHeaders = np.array(data[0])
    dataAr = np.array(data[1:])
    df1 = pd.DataFrame(dataAr, columns=dataHeaders)
    df1 = df1[df1.columns.dropna()]
    return df1


# Function: Read in input excels and convert to df
def read_excel_convert_to_df(file):
    workbook = openpyxl.load_workbook(file, read_only=True)
    worksheet = workbook.worksheets[0]
    data = list(worksheet.values)
    workbook.close()
    df1 = createDf(data)
    return df1


# Function: Combined all excels in a particular folder
def combine_excels(list_excel_names):
    df_list = []
    for excel in list_excel_names:
        print(excel)
        df = read_excel_convert_to_df(excel)
        df_list.append(_df)
    df_out = pd.concat(df_list).reset_index(drop=False)
    return df_out


def create_matching_matrix(column1, column2):
    # Initialize an empty matrix to store scores
    num_worker_rows = len(column1)
    num_work_rows = len(column2)
    matrix = [[0] * num_work_rows for _ in range(num_worker_rows)]

    # Iterate over each row in the first column
    for i, target_list in enumerate(column1):
        # Iterate over each row in the second column
        for j, approach_list in enumerate(column2):
            # Check if both lists are not None and not empty
            if target_list and approach_list:
                # Convert both lists of tokens to sets for efficient membership testing
                target_words = set(target_list.lower().split(','))
                approach_words = set(approach_list.lower().split(','))
                # Check if there's any intersection between the sets
                if target_words.intersection(approach_words):
                    matrix[i][j] = 1
    # Convert the matrix to a pandas DataFrame
    matrix_df = pd.DataFrame(matrix)
    return matrix_df


def normalize_scores(matrix_df):
    # Apply Min-Max normalization to each element of the matrix DataFrame
    normalized_matrix_df = (  ( matrix_df - matrix_df.min().min() ) / (matrix_df.max().max() - matrix_df.min().min())  ) + 0.0001
    return normalized_matrix_df



# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sentence_model(sentences):
    # Load model from local folder
    tokenizer = AutoTokenizer.from_pretrained('../src/models/custom_sentence-transformer/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('../src/models/custom_sentence-transformer/all-MiniLM-L6-v2')
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

    
# =============================================================================
# Script B functions
# =============================================================================

def concat_strings(series):
    return ', '.join(series)


def get_integer_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value  # Return the integer value if input is valid
        except ValueError:
            print("Error: Please enter an integer.")
            
            
        



