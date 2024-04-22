import numpy as np
import pandas as pd
import datetime
import pathlib
import os
import traceback
import openpyxl
import tkinter, tkinter.messagebox, tkinter.filedialog
import torch
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity

# Parameters
# this corresponds to the sheet names in your excel database
sheet_names = ['Worker_Attributes', 'Work_Attributes', 'Weightage']
inputfolder = "input"



try:
    # Get timestamp
    now = datetime.datetime.now()
    var_tsYmd = now.strftime("%Y_%m_%d")
    var_tsSql = now.strftime("%Y-%m-%d %H:%M:%S")
    var_tsWin = now.strftime("%Y %m %d_%H%M%S")
    var_username = pathlib.Path().home().name
    print("Date-time now: " + var_tsSql)
    print("Username: " + var_username)
    print("...Started processing. Please wait until program says it is done.")
    
    #------Find files and folder--------
    tkinter.Tk().withdraw()
    tkinter.messagebox.showinfo("Instructions", "Please choose the working folder containing the input & output subfolders")
    
    working_folder = tkinter.filedialog.askdirectory()
    working_folder_path = pathlib.Path(working_folder)
    
    utils_folder_path = working_folder_path.parent/ "src"
    os.chdir(utils_folder_path)
    import utils
    os.chdir(working_folder_path)
    
    print(r"Folder chosen: '" + str(working_folder_path) +r"'")
    
    
    # ------------Start of Code------------
    
    inputfolder_Path = pathlib.Path(inputfolder)
    
    
    # Extract the different tables from excel dataset
    inputfolder_excel_files = inputfolder_Path.glob('[!~]*.xlsx')
    first_excel_file = next(inputfolder_excel_files, None)
    workbook = openpyxl.load_workbook(first_excel_file)
    
    dfs_excel = {}  # Use a dictionary to store DataFrames for each sheet

    for sheet_name in sheet_names:
        try:
            x_worksheet = workbook[sheet_name]
            x_table = x_worksheet.tables[sheet_name]
            x_table_range = x_table.ref
            x_data = []
    
            for row in x_worksheet[x_table_range]:
                x_data.append([cell.value for cell in row])
    
            x_headers = x_data[0]
            dfs_excel[sheet_name] = pd.DataFrame(x_data[1:], columns=x_headers)
    
        except KeyError as e:
            print(f"Error: {e}. Please check that the table exists or is in the correct table name for sheet {sheet_name}.")	
    
    
    # Define the tables
    Worker_Attributes = dfs_excel['Worker_Attributes']
    Work_Attributes = dfs_excel['Work_Attributes']
    Weightage = dfs_excel['Weightage']
    
    # Filter the tables for those included for allocation
    Worker_Attributes = Worker_Attributes[Worker_Attributes['Included_for_assignment'] == 1].reset_index().drop(columns='index')
    Work_Attributes = Work_Attributes[Work_Attributes['Included_for_assignment'] == 1].reset_index().drop(columns='index')
    
    # For each Worker, split each word in expertise, get each embeds
    print("processing embeddings for Workers expertise...")
    Worker_Attributes_split = Worker_Attributes['Worker Expertise'].str.split(',')
    
    expertise_embeddings_list = []
    for expertise_list in Worker_Attributes_split:
        tokens_embeddings=[]
        for token in expertise_list:
            token_embeds = utils.sentence_model(token)
            tokens_embeddings.append(token_embeds)
        # Convert the list of token embeddings to a PyTorch tensor
        tokens_embeddings_tensor = torch.stack(tokens_embeddings)
        expertise_embeddings_list.append(tokens_embeddings_tensor)   
    print("embeddings for Workers expertise completed")

    # Split each word in Work's keywords, get each embeds
    print("processing embeddings for Work Project Keywords...")
    Work_Keywords_split = Work_Attributes['Project Keywords'].str.split(',')
    keyword_embeddings_list = []
    for keywords_list in Work_Keywords_split:
        tokens_embeddings=[]
        for token in keywords_list:
            token_embeds = utils.sentence_model(token)
            tokens_embeddings.append(token_embeds)
        # Convert the list of token embeddings to a PyTorch tensor
        tokens_embeddings_tensor = torch.stack(tokens_embeddings)
        keyword_embeddings_list.append(tokens_embeddings_tensor)        
    print("embeddings for Work keywords completed")
    
# =============================================================================
# Work keywords and Worker expertise scoring

    # Compute 'similarity score' and 'top keywords combination' for each Work / Worker keypair
    print("processing similarity scores between Work keywords and Worker Expertise...")
    max_similarities = []
    max_similarities_keywords = []
    
    for combination_index, ((Work_idx, Work), (Worker_idx, Worker)) \
        in enumerate(product(enumerate(keyword_embeddings_list), enumerate(expertise_embeddings_list))):
            
        max_score = float('-inf')
        max_Work_keyword = None
        max_Worker_keyword = None
        
        for Worker_token_idx, Worker_token_emb in enumerate(Worker):
            for Work_token_idx, Work_token_emb in enumerate(Work):
                score = cosine_similarity(Work_token_emb, Worker_token_emb)[0][0]
                if score > max_score:
                    max_score = score
                    max_Work_keyword = Work_Keywords_split[Work_idx][Work_token_idx]
                    max_Worker_keyword = Worker_Attributes_split[Worker_idx][Worker_token_idx]
        max_similarities.append(max_score)
        max_similarities_keywords.append((max_Work_keyword, max_Worker_keyword))
    
    # Convert to dataframes
    num_Work = len(keyword_embeddings_list)
    num_Workers = len(expertise_embeddings_list)
    total_elements = num_Work * num_Workers
    max_similarities_matrix = torch.tensor(max_similarities).reshape(num_Work, num_Workers)
    expertise_Work_keyword_score_df = pd.DataFrame(max_similarities_matrix.numpy()).T
    expertise_keywords_normalized = utils.normalize_scores(expertise_Work_keyword_score_df)
    max_similarities_keywords_matrix = [tuple(max_similarities_keywords[i:i+num_Workers]) for i in range(0, total_elements, num_Workers)]
    expertise_Work_keyword_df = pd.DataFrame(max_similarities_keywords_matrix).T
    print("similarity scores between Work keywords and Worker Expertise completed")
 
# =============================================================================
# Project Title and Worker Expertise

    # Get Embeddings for Work title
    title_embedding = utils.sentence_model(Work_Attributes['Project Title'].tolist())

    # Convert title_embedding to a list of tensors with shape [1, 384]
    title_embedding_list = [title_embedding[i].unsqueeze(0) for i in range(title_embedding.shape[0])]

    # Compute 'similarity score' and 'top Worker keywords' for each Worker/Title pair
    print("processing similarity scores between Work title and Worker Expertise...")
    max_similarities = []
    max_similarities_keywords = []
    
    for combination_index, ((title_idx, title), (Worker_idx, Worker)) \
        in enumerate(product(enumerate(title_embedding_list), enumerate(expertise_embeddings_list))):
            max_score = float('-inf')
            max_Worker_keyword = None
        
            for Worker_token_idx, Worker_token_emb in enumerate(Worker):
                score = cosine_similarity(title, Worker_token_emb)[0][0]
                if score > max_score:
                    max_score = score
                    max_Worker_keyword = Worker_Attributes_split[Worker_idx][Worker_token_idx]
            max_similarities.append(max_score)
            max_similarities_keywords.append(max_Worker_keyword)

    # Convert to dataframes and normalize
    num_Work = len(title_embedding_list)
    num_Workers = len(expertise_embeddings_list)
    total_elements = num_Work * num_Workers
    max_similarities_matrix = torch.tensor(max_similarities).reshape(num_Work, num_Workers)
    expertise_title_score_df = pd.DataFrame(max_similarities_matrix.numpy()).T
    expertise_title_normalized = utils.normalize_scores(expertise_title_score_df)
    max_similarities_keywords_matrix = [tuple(max_similarities_keywords[i:i+num_Workers]) for i in range(0, total_elements, num_Workers)]
    expertise_title_keyword_df = pd.DataFrame(max_similarities_keywords_matrix).T
    print("similarity scores between Work title and Worker Expertise completed")

    
    # Use defined function to create the matrixes for exact match features
    print("computing scores for exact match features...")
    institution_matrix = utils.create_matching_matrix(Worker_Attributes['Institution'], Work_Attributes['Project Institute'])
    Target_Project_matrix = utils.create_matching_matrix(Worker_Attributes['Preference'], Work_Attributes['Type_of_Project'])
    Specialty_matrix = utils.create_matching_matrix(Worker_Attributes['Specialty'], Work_Attributes['Specialty'])
    
    # For institution matrix, Convert >0 values to 0, and 0 values to 1 (must variable)
    institution_matrix_transformed = pd.DataFrame(np.where(institution_matrix > 0, 0, 1))
    print("scores for exact match features completed")
    
    # multiply non-must's (include exact match) features by weights
    print("aggregating scores...")
    expertise_keywords_weighted = expertise_keywords_normalized * Weightage.loc[Weightage['Field'] == 'Expertise_keywords']['Weight'].values[0]
    expertise_title_weighted = expertise_title_normalized * Weightage.loc[Weightage['Field'] == 'Expertise_title']['Weight'].values[0]
    
    Target_Project_matrix_weighted = Target_Project_matrix * Weightage.loc[Weightage['Field'] == 'Type_of_Project']['Weight'].values[0]
    Specialty_matrix_weighted = Specialty_matrix * Weightage.loc[Weightage['Field'] == 'Specialty']['Weight'].values[0]
    
    # sum up all non-must features
    sum_non_must_matrix = (expertise_keywords_weighted + expertise_title_weighted +\
                           Target_Project_matrix_weighted + Specialty_matrix_weighted)
    
    # multiply summed 'non-must' matrix with 'must' matrix (i.e., institution)
    output_matrix = sum_non_must_matrix * institution_matrix_transformed
    print("aggregated scores computed")
    
    # assign Work names to columns
    Work_names = Work_Attributes['Project Number']
    output_matrix.columns = Work_names
    institution_matrix_transformed.columns = Work_names
    Target_Project_matrix.columns = Work_names
    Specialty_matrix.columns = Work_names
    expertise_title_normalized.columns = Work_names
    expertise_keywords_normalized.columns = Work_names
    expertise_Work_keyword_df.columns = Work_names
    expertise_title_keyword_df.columns = Work_names

    # assign Worker names to index
    Worker_names = Worker_Attributes['Name of Worker']
    output_matrix.index = Worker_names
    institution_matrix_transformed.index = Worker_names
    Target_Project_matrix.index = Worker_names
    Specialty_matrix.index = Worker_names
    expertise_title_normalized.index = Worker_names
    expertise_keywords_normalized.index = Worker_names
    expertise_Work_keyword_df.index = Worker_names
    expertise_title_keyword_df.index = Worker_names

    # Create a Pandas Excel writer using ExcelWriter
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_path_filename = f'outputs/Part_A_outputs_{formatted_datetime}.xlsx'
    
    with pd.ExcelWriter(output_path_filename) as writer:
        output_matrix.to_excel(writer, sheet_name='output_matrix', index=True)
        institution_matrix_transformed.to_excel(writer, sheet_name='institution', index=True)
        Target_Project_matrix.to_excel(writer, sheet_name='Target_project', index=True)
        Specialty_matrix.to_excel(writer, sheet_name='Specialty', index=True)
        expertise_keywords_normalized.to_excel(writer, sheet_name='expertise_keywords_score', index=True)
        expertise_Work_keyword_df.to_excel(writer, sheet_name='expertise_keywords_df', index=True)
        expertise_title_normalized.to_excel(writer, sheet_name='expertise_title_score', index=True)
        expertise_title_keyword_df.to_excel(writer, sheet_name='expertise_title_df', index=True)
        
        
except:
	err = str(traceback.format_exc())
	print(err)

finally:
	print("Script A IS DONE...Check Output folder..Press any key twice to continue")
	input()







