import pandas as pd
import datetime
import pathlib
import os
import traceback
import openpyxl
import tkinter.messagebox, tkinter.filedialog
from pulp import *
import math

# Paramters
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

    
    #------Find input from part 1 output file --------
    tkinter.Tk().withdraw()
    tkinter.messagebox.showinfo("Instructions", "Please choose the output_A matrix file")
    path_df_in = tkinter.filedialog.askopenfilename()

    utils_folder_path = pathlib.Path(path_df_in).parent.parent.parent/ "src"
    os.chdir(utils_folder_path)
    import utils
    working_folder = utils_folder_path.parent/ "data"
    os.chdir(working_folder)


    # Read part A output file
    df_in = pd.read_excel(path_df_in)
    df_in.rename(columns={'Name of Worker':'Project'}, inplace=True)
    df_in.set_index('Project',inplace=True)
    df_score = df_in.T
    
    # declare value for minimum number of workers for each project
    minimum_worker = utils.get_integer_input("Enter the exact number of workers required for each project: ")
    print(f"Note: You have entered {minimum_worker} for the number of workers required for each project")
    no_of_projects = len(df_score)
    no_of_assignments = minimum_worker * no_of_projects
    
    
    # order from top to btm the rank of workers for each project
    sorted_df_project_level = pd.DataFrame()
    for col in df_in.columns:
        sorted_values = df_in[col].sort_values(ascending=False)
        sorted_workers = [f"{name}, {score:.6f}" for name, score in zip(sorted_values.index, sorted_values)]
        sorted_df_project_level[col] = sorted_workers
    sorted_df_project_level = sorted_df_project_level.T
    new_columns = {old_col: f'Rank {int(old_col)+1}' for old_col in sorted_df_project_level.columns}
    sorted_df_project_level.rename(columns=new_columns, inplace = True)
    
    # order from top to btm the rank of projects for each worker
    df_worker_level = df_in.T
    sorted_df_worker_level = pd.DataFrame()
    for col in df_worker_level.columns:
        sorted_values = df_worker_level[col].sort_values(ascending=False)
        sorted_projects = [f"{project}, {score:.6f}" for project, score in zip(sorted_values.index, sorted_values)]
        sorted_df_worker_level[col] = sorted_projects
    sorted_df_worker_level = sorted_df_worker_level.T
    new_columns = {old_col: f'Rank {int(old_col)+1}' for old_col in sorted_df_worker_level.columns}
    sorted_df_worker_level.rename(columns=new_columns, inplace = True)
    
    
    # Locate original input file
    inputfolder_Path = pathlib.Path(inputfolder)
    
    # Extract the different tables from the first workRA_dev_worksheet in input folder
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
            print(f"Error: {e}. Please check that the table exists or is in the \
                  correct table name for sheet {sheet_name}.")	
    
    # Define the tables
    Worker_Attributes = dfs_excel['Worker_Attributes']
    
    # Filter the tables for those included for allocation
    Worker_Attributes = Worker_Attributes[Worker_Attributes['Included_for_assignment'] == 1].reset_index().drop(columns = "index")
    
    # Filter for Staff and capacity
    staff_max_prof = Worker_Attributes[['Name of Worker','Max capacity', 'Professor']]
    staff_max_prof.columns = ['names', 'Max', 'Professor']
    staff_max_prof.set_index('names',inplace=True)
    staff_max_prof.index.name = None
    
    # Calculating capacity
    non_Prof_count = len(staff_max_prof[staff_max_prof['Professor']==0])
    non_Prof_total_capacity = staff_max_prof[staff_max_prof['Professor']==0]['Max'].sum()
    Prof_count = len(staff_max_prof[staff_max_prof['Professor']==1])
    Prof_total_capacity = staff_max_prof[staff_max_prof['Professor']==1]['Max'].sum()
    total_capacity = non_Prof_total_capacity + Prof_total_capacity
    
    # number of Prof worker per work (hardcoded 1)
    Prof_required_per_work = 1
    
    # number of work that can be assigned based on number of Prof workers and their capacity
    Prof_work_limit = math.floor(Prof_total_capacity / Prof_required_per_work)
    
    
    # Check for Prof capacity requirements
    print(F"Note: You have {Prof_count} Professors with a total capacity of {Prof_total_capacity}")
    print(F"Note: Given that you require {Prof_required_per_work} Professor for each project, you can assign up to {Prof_work_limit} projects at most")
      
    if no_of_projects > Prof_work_limit:
        while True:
            user_input = input(f"\n*** Warning ***: There are {no_of_projects} works to assign, you do not have enough Professor capacity, an optimal solution will not be possible, do you wish to continue? (y/n): ")
            if user_input.lower() == 'n':
                print(f"You selected {user_input}, the script will end here and an output will not be provided.")
                exit()
            elif user_input.lower() == 'y':
                print(f"You selected {user_input}, the script will continue and an output will still be provided.\n\n")
                break  # Break out of the loop if valid input is provided
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    else:
        print(f"Note: There is sufficient Professor capacity to assign to {no_of_projects} work.\n\n")


    
    # number of non-Prof required for each work given the number of Prof required defined above
    non_Prof_required_per_work = minimum_worker - Prof_required_per_work
    
    # number of work that can be assigned based on number of non-Prof workers and their capacity
    # condition such that if non_Prof_required_per_work is 0, then non_Prof_work_limit should be a large number
    if non_Prof_required_per_work == 0:
        non_Prof_work_limit = 999
        print(F"Note: You have {non_Prof_count} non-Prof workers with a total capacity of {non_Prof_total_capacity}")
        print(F"Note: Given that you require {non_Prof_required_per_work} non-Prof worker for each work, there is no limit to the number of project you can assign")
    else:
        non_Prof_work_limit = math.floor(non_Prof_total_capacity / non_Prof_required_per_work)
        print(F"Note: You have {non_Prof_count} non-Prof workers with a total capacity of {non_Prof_total_capacity}")
        print(F"Note: Given that you require {non_Prof_required_per_work} non-Prof worker for each work, you can assign up to {non_Prof_work_limit} projects at most")
   

    if no_of_projects > non_Prof_work_limit:
        user_input = input(f"\n*** Warning ***: There are {no_of_projects} projects to assign, you do not have enough non_Prof worker capacity, an optimal solution will not be possible, do you wish to continue? (y/n): ")
        if user_input.lower() == 'n':
            print(f"you selected {user_input}, the script will end here and an output will not be provided.")
            exit()
        else:
            print(f"you selected {user_input}, the script will continue and an output will still be provided.\n\n")
    else:
        print(f"Note: There is sufficient non_Prof worker capacity to assign to {no_of_projects} works.\n\n")


    
#   Solver
# =============================================================================
    # Create a LP maximization problem
    prob = LpProblem("worker_Assignment", LpMaximize)
    
    # Define variables
    workers = df_score.columns.tolist()  # List of workers
    projects = df_score.index.tolist()  # List of projects
    
    # Binary variables indicating whether worker i is assigned to paper j
    x = LpVariable.dicts("Assign", (workers, projects), cat='Binary')
    
    # Create a dictionary to store scores between workers and project
    scores = {(i, j): df_score.loc[j, i] for i in workers for j in projects}
    
    # Objective function: maximize the total score of assignments
    prob += lpSum(scores[i, j] * x[i][j] for i in workers for j in projects if scores.get((i, j), 0) != 0)
        
    # Constraints: each worker can review an X number of projects
    for i in workers:
        prob += lpSum(x[i][j] for j in projects) <= staff_max_prof.loc[i, 'Max']

    # Constraints: each project must have exactly X number of workers
    for j in projects:
        prob += lpSum(x[i][j] for i in workers) == minimum_worker

    # Constraints: each project need to have 1 non-Prof worker (Prof status = 0)
    for j in projects:
        prob += lpSum(x[i][j] for i in workers if staff_max_prof.loc[i, 'Professor'] == 0) == 1

    # Fairness constraint: limit the difference in workload among workers
    # declare value for fairness_tolerance
    fairness_tolerance = utils.get_integer_input("What is the maximum difference in the number of projects assigned between any two workers?: ")
    
    # Introduce auxiliary variables for maximum and minimum assignments
    max_assigned = LpVariable("Max_Assigned", lowBound=0, cat='Integer')
    min_assigned = LpVariable("Min_Assigned", lowBound=0, cat='Integer')
    
    # Constraint to update maximum and minimum assignments
    for i in workers:
        prob += lpSum(x[i][j] for j in projects) <= max_assigned
        prob += lpSum(x[i][j] for j in projects) >= min_assigned
    
    # Fairness constraint: limit the difference in workload among workers
    prob += max_assigned - min_assigned <= fairness_tolerance

    # Solve the problem
    prob.solve()


# =============================================================================
    # Ask the user for input to decide whether to iterate or skip when solution is not optimal
    
    if prob.status != LpStatusOptimal:
        user_input = input("Initial solution is not optimal. Do you want to iterate through and modify each constraint (1 at a time) until an optimal solution is reached? (y/n): ")
        
        if user_input.lower() == 'y':
            # Flag to track if infeasibility is due to multiple constraints
            multiple_constraints = False
        
            # Iterate to identify infeasible constraints
            infeasible_constraints = []
            for constraint in prob.constraints.values():
                constraint.sense = -constraint.sense  # Change constraint sense (temporarily relax)
                prob.solve()
                if prob.status == LpStatusOptimal:
                    infeasible_constraints.append(constraint)
                constraint.sense = -constraint.sense  # Restore original constraint sense
        
            if len(infeasible_constraints) > 0:
                print("Constraints that resulted in infeasible initial solution: \n\n")
                for constraint in infeasible_constraints:
                    print(constraint)
            else:
                print("Modifying the constraints (1 at a time) did not result in an optimal solution...\n\n")  
        else:
            print("Initial solution is not optimal and user has opted to skip iterating through the constraints. An output will still be provided.\n\n")
    else:
        print("Initial solution is optimal, skipping iteration...\n\n")
    
    # Print the results
    print("Status:", LpStatus[prob.status])
    print("Total Score of Assignments:", value(prob.objective))
    
# =============================================================================
    

    # Format the assignments into a dataframe
    assignment_results = []
    for project in projects:
        assigned_workers = [i for i in workers if x[i][project].value() == 1]
        for worker in assigned_workers:
            score = scores.get((worker, project), 0)  # Get the score from the scores dictionary
            assignment_results.append({'Project': project, 'worker': worker, 'Score': score})

    # Convert the dictionary results into a DataFrame
    df_assignment = pd.DataFrame(assignment_results).sort_values('Project')
    
    # Group by 'Project' and apply the custom aggregation function
    df_project_level = df_assignment.groupby('Project')['worker'].agg(utils.concat_strings).reset_index()
    df_project_level['worker_count'] = df_project_level['worker'].apply(lambda x: len(x.split(', ')))

    # Group by 'worker' and apply the custom aggregation function
    df_worker_level = df_assignment.groupby('worker')['Project'].agg(utils.concat_strings).reset_index()
    df_worker_level['Project_count'] = df_worker_level['Project'].apply(lambda x: len(x.split(', ')))
    df_worker_level = df_worker_level.merge(dfs_excel['Worker_Attributes'], how='left', left_on ='worker', right_on= 'Name of Worker')
    df_worker_level = df_worker_level[['worker', 'Project', 'Project_count', 'Max capacity']]
    
    # Pivot table to have overview
    df_overview = df_assignment.pivot_table(index='Project', columns='worker', values='Score')
    
    # Merge with orignial tables to get full details
    worker_df = dfs_excel['Worker_Attributes'].add_prefix('worker_')
    df_full_details_1 = df_assignment.merge(worker_df, how='left', left_on ='worker', right_on= 'worker_Name of Worker')
    
    work_df = dfs_excel['Work_Attributes'].add_prefix('work_')
    df_full_details_2 = df_full_details_1.merge(work_df, how='left', left_on ='Project', right_on= 'work_Project Number')
    
    # Define a list of columns to keep
    columns_to_keep = ['Project', 'worker', 'Score', 
                       'worker_Institution','work_Project Institute',
                       'worker_Worker Expertise', 'work_Project Title', 'work_Project Keywords',
                       'worker_Preference','work_Type_of_Project',
                       'worker_Specialty', 'work_Specialty',
                       'worker_Professor', 'worker_Remarks'
                       ]


    # Select columns to keep using .loc accessor
    df_full_details = df_full_details_2.loc[:, columns_to_keep]
    
    # Create a Pandas Excel writer using ExcelWriter
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_path_filename = f'outputs/Part_B_outputs_{formatted_datetime}.xlsx'
    
    with pd.ExcelWriter(output_path_filename) as writer:
        df_overview.to_excel(writer, sheet_name='df_overview', index=True)
        df_assignment.to_excel(writer, sheet_name='df_assignment', index=True)
        df_project_level.to_excel(writer, sheet_name='df_project_level', index=True)
        sorted_df_project_level.to_excel(writer, sheet_name='candidates_df_project_level', index=True)
        df_worker_level.to_excel(writer, sheet_name='df_worker_level', index=True)
        sorted_df_worker_level.to_excel(writer, sheet_name='candidates_df_worker_level', index=True)
        df_full_details.to_excel(writer, sheet_name='df_assign_full_details', index=True)
        
        
        
except:
	err = str(traceback.format_exc())
	print(err)

finally:
	print("Script B IS DONE...Check Output folder..Press any key twice to continue")
	input()

