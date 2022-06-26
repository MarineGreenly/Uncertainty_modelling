
from pyparsing import dict_of
import compute_base_CV
import Monte_Carlo_Sum
import pandas as pd
import os



def sensitivity_analysis_methodologies_assumptions(value):
    # Reads results tables from ADEME uncertainty analysis, and computes statitics on uncertainty metrics depending on 
    # assumption on data quality and methodologies
    directory = ("Sensitivity data quality and methodology")
    folder_exist = os.path.isdir(directory)
    if not folder_exist:
        os.makedirs("Sensitivity data quality and methodology")

    list_cases = ["worst", "average", "best"]
    total_probability = []
    for assumption in list_cases:
        list_probability = []
        for methodology in list_cases:
            print(assumption, methodology)
            statistics = pd.read_csv("ADEME_uncertainty_assessement/statistics_ADEME_uncertainty_assessment_"+assumption+"_"+methodology+".csv")
            print(statistics)
            p = float(statistics.loc[(statistics["Unnamed: 0"]=="mean"),value])
            list_probability.append(p)
        total_probability.append(list_probability)
    dataframe = pd.DataFrame(total_probability)
    dataframe.to_csv("./Sensitivity data quality and methodology/"+value+".csv", encoding='utf-8')
    print(total_probability)

# statistics = pd.read_csv("ADEME_uncertainty_assessement/statistics_ADEME_uncertainty_assessment_best_best.csv")
# for column in statistics.columns[2:]:
#     sensitivity_analysis_methodologies_assumptions(column)
#     print("Done "+ column)



def get_list_variables():
    #Attributes id to all variable assumptions made when building the uncertainty model

    directory = ("Sensitivity variables")
    folder_exist = os.path.isdir(directory)
    if not folder_exist:
        os.makedirs("Sensitivity variables")

    dict_variables = {}
    methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors = compute_base_CV.import_model_parameters()
    list_inputs = base_uncertainty_factors, base_uncertainty_data, description_matrix_data, description_matrix_factor
    index = 0
    for e, input in enumerate(list_inputs):
        print(e)
        if e < 2:
            for i, variable in input.iterrows():
                    dict_variables[variable[0]] = index
                    index+=1
        else:
            for variable in input.ravel().tolist():
                dict_variables[variable] = index
                index+=1
    dataframe = pd.DataFrame(dict_variables, index=[1]).transpose()
    dataframe.to_csv("Sensitivity variables/index_variables.csv")

# get_list_variables()


def sensitivity_analysis_parameters(list_var):
    # Takes all variable identified in the previous function
    # Transformed them 1 by 1 to 0 or 1
    # Computes resulting distance in final uncertainty metric for all ADEME assessments
    # Stores results

    print("Monte Carlo precision")
    list_dataframe = []

    # Computes distance without change to assess natural variability of Monte Carlo analyis
    for k in range(0,2):
        print(k)
        methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors = compute_base_CV.import_model_parameters(False)
        results = compute_base_CV.calculations_underlognormal_assumption(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors, False)
        ADEME_uncertainty_assessment = Monte_Carlo_Sum.assess_ADEME(100000, "average", "average", sensitivity_analysis = True)
        list_dataframe.append(ADEME_uncertainty_assessment) 
    dataframe = pd.concat(list_dataframe)
    dataframe.groupby("id").min().to_csv("Sensitivity variables/Sensitivity_results_min.csv")
    dataframe.groupby("id").max().to_csv("Sensitivity variables/Sensitivity_results_max.csv")
    dataframe = dataframe[["Most probable total value (tCO2e/collaborator)", "Probability E1>=E2",	"Probability E2>=E3",	"Probability E3>=E4",	"Probability E4>=E5",	"Probability good order on most important declared scopes","Uncertainty measure","id"]]
    dataframe_distances = dataframe.groupby("id").max()-dataframe.groupby("id").min()
    dataframe_distances.describe().to_csv("Sensitivity variables/Sensitivity_analysis_distance.csv")

    # Computes distances on final GHG assessments for parameters set to 0 and 1
    for var in list_var:
        print()
        print("variable : ", var)
        list_dataframe = []
        for i in range(0,2):
            print("iteration : ",i)
            # print("Call 1")
            methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors = compute_base_CV.import_model_parameters(True, var,i*100)
            # print("import done")
            results_average = compute_base_CV.calculations_underlognormal_assumption(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors, True)
            results_worst = compute_base_CV.calculations_worst_case(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors, True)
            results_best = compute_base_CV.calculations_best_case(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors, True)
            # print(results)
            for index,results in enumerate([results_average, results_best, results_worst]):
                print(index)
                ADEME_uncertainty_assessment = Monte_Carlo_Sum.assess_ADEME(100000, "average", "average", sensitivity_analysis = True ,results_var = results, var=var)
                ADEME_uncertainty_assessment["Assumption"] = index
                list_dataframe.append(ADEME_uncertainty_assessment) 

        # Computes and stores results for each variable
        dataframe = pd.concat(list_dataframe)
        print(dataframe.describe())
        dataframe.groupby(["id", "Assumption"]).min().to_csv("Sensitivity variables/Sensitivity_results_min"+str(var)+".csv")
        dataframe.groupby(["id", "Assumption"]).max().to_csv("Sensitivity variables/Sensitivity_results_max"+str(var)+".csv")
        dataframe = dataframe[["Most probable total value (tCO2e/collaborator)", "Probability E1>=E2",	"Probability E2>=E3",	"Probability E3>=E4",	"Probability E4>=E5",	"Probability good order on most important declared scopes","Uncertainty measure","id", "Assumption"]]
        dataframe_distances = dataframe.groupby(["id", "Assumption"]).max()-dataframe.groupby(["id", "Assumption"]).min()
        dataframe_distances.describe().to_csv("Sensitivity variables/Sensitivity_analysis_distance"+str(var)+".csv")

# dataframe = pd.read_csv("Sensitivity variables/index_variables.csv")
# sensitivity_analysis_parameters([i for i in range(0,len(dataframe))])


def group_sensitivity_analysis_results():
    # Groups the results computed above, computes the distances and sets the structure of the dataframe.
    # Save the final results
    reference_dataframe = pd.read_csv("Sensitivity variables/index_variables.csv")
    total_var = len(reference_dataframe)
    list_dataframe = []
    for var in range(0,total_var):
        dataframe_min = pd.read_csv("Sensitivity variables/Sensitivity_results_min"+str(var)+".csv")
        dataframe_min = dataframe_min[["Most probable total value (tCO2e/collaborator)", "Probability E1>=E2",	"Probability E2>=E3",	"Probability E3>=E4",	"Probability E4>=E5",	"Probability good order on most important declared scopes","Uncertainty measure","id", "Assumption"]]
        dataframe_max = pd.read_csv("Sensitivity variables/Sensitivity_results_max"+str(var)+".csv")[["Most probable total value (tCO2e/collaborator)", "Probability E1>=E2",	"Probability E2>=E3",	"Probability E3>=E4",	"Probability E4>=E5",	"Probability good order on most important declared scopes","Uncertainty measure","id", "Assumption"]]
        dataframe_max = dataframe_max[["Most probable total value (tCO2e/collaborator)", "Probability E1>=E2",	"Probability E2>=E3",	"Probability E3>=E4",	"Probability E4>=E5",	"Probability good order on most important declared scopes","Uncertainty measure","id", "Assumption"]]
        
        dataframe_max.set_index(["id", "Assumption"], inplace=True)
        dataframe_min.set_index(["id", "Assumption"], inplace=True)
         
        # print(dataframe_max)
        
        dataframe_distances = dataframe_max-dataframe_min
        
        dataframe = dataframe_distances.groupby("Assumption").mean()
        dataframe["var"] = var
        dataframe["variable_description"] = reference_dataframe.loc[reference_dataframe["1"]==var, "Unnamed: 0"]
        list_dataframe.append(dataframe)
    list_dataframe = pd.concat(list_dataframe)
    list_dataframe.to_csv("Sensitivity variables/Comparison_of_entry_variables.csv")

# group_sensitivity_analysis_results()




