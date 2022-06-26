from email.mime import base
from typing import final
import pandas as pd
import numpy as np
import math
import scipy
import os


def import_model_parameters(sensivity_analysis_initial = False, var = 0, k = 0, default=True):
    #Read model parameters from "Entry data" folder
    methodologies = pd.read_csv("./Entry data/review_methodology - Calculation methodologies.csv",index_col=False)
    base_uncertainty_factors = pd.read_csv("./Entry data/review_methodology - Factors.csv", index_col=False)

    if default:
        base_uncertainty_data = pd.read_csv("./Entry data/review_methodology - Data.csv", index_col=False)
    else:
        base_uncertainty_data = pd.read_csv("./Entry data/Entry data client/review_methodology - Data.csv", index_col=False)

    matrix_data = pd.read_csv("./Entry data/review_methodology - Matrix Data.csv",header=0)
    list_categories_data = matrix_data['Category'].to_numpy()
    matrix_data.drop(['Category'], axis=1, inplace=True)
    matrix_data = matrix_data.to_numpy()

    matrix_factor = pd.read_csv("./Entry data/review_methodology - Matrix Factor.csv", header=0)
    list_categories_factors = matrix_factor['Category'].to_numpy()
    matrix_factor.drop(['Category'], axis=1, inplace=True)
    matrix_factor = matrix_factor.to_numpy()

    description_matrix_data = pd.read_csv("./Entry data/review_methodology - Description Matrix Data.csv",header=0)
    description_matrix_data.drop(['Category'], axis=1, inplace=True)
    description_matrix_data = description_matrix_data.to_numpy()

    description_matrix_factor = pd.read_csv("./Entry data/review_methodology - Description Matrix Factor.csv", header=0)
    description_matrix_factor.drop(['Category'], axis=1, inplace=True)
    description_matrix_factor = description_matrix_factor.to_numpy()

    # If performing sensitivity analysis, change randomly from +50% to - 50% the variable var in the model parameters
    if sensivity_analysis_initial:
        # rv = scipy.stats.uniform(loc=0,scale=2)
        list_inputs = base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor
        index = 0
        for e, input in enumerate(list_inputs):
            # print(input)
            if e < 2:
                # print(input.head())
                for i, variable in input.iterrows():
                    if index == var:
                        initial_value = input.loc[i,"Default base uncertainty"]
                        input.loc[i,"Default base uncertainty"] = k
                        final_value = input.loc[i,"Default base uncertainty"]
                        print("Variable  modified !", index," ", initial_value, " ", final_value)
                        index+=1
                    else:
                        index+=1
            else:
                for variable in input.ravel().tolist():
                    # print(index)
                    if index == var:
                        if type(variable) == str:
                            index+=1
                            continue
                        else:
                            initial_value = variable
                            variable = k
                            final_value = variable
                            print("Variable modified ! ", index, " ", initial_value, " ", final_value)
                            index+=1
                    else:
                        index+=1
    
    # print(base_uncertainty_factors.describe())
    # print(methodologies.describe())
    # print(matrix_data)
    # print(matrix_factor)
    # print(description_matrix_data)
    # print(description_matrix_factor)

    return methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors

import_model_parameters(sensivity_analysis_initial = True, var = 0, k = 0, default=True)

def calculations_underlognormal_assumption(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors, sensitivity_analysis=False ):
    # Compute base coefficients of variation for each available methodology under assumption most likely quality of data
    for column in methodologies.columns:
        if "Factor" in column:
            methodologies[column] = methodologies[column].astype('str')
            fit = base_uncertainty_factors.rename({"Factor":column, "Default base uncertainty": "Default base uncertainty "+column, "Reliability" : "Reliability "+column, "Temporal correlation" : "Temporal correlation "+column, "Geographical correlation": "Geographical correlation "+column, "Product correlation": "Product correlation "+column }, axis =1)
            for index, item in fit.iterrows():
                if str(item[column])=="nan":
                    for i in range(0,4):
                        fit.at[index,list_categories_factors[i] + " "+ column + " value"] = 0
                    continue
                else:
                    factors = []
                    for i in range(0,4):
                        # print(list_categories_factors[i] + " "+ column, matrix_factor[i,int(item[list_categories_factors[i]+" "+column])])
                        factors.append(matrix_factor[i,int(item[list_categories_factors[i]+" "+column])])
                        fit.at[index,list_categories_factors[i] + " "+ column]= factors[-1]
                    fit.at[index,"Final uncertainty "+column] = math.sqrt(np.prod([(factors[i]+1)**2 for i in range(0,len(factors))])*(float(item["Default base uncertainty "+column])+1)**2-1)
            fit[column] = fit[column].astype('str')
            methodologies = pd.merge(methodologies,fit[[column, "Default base uncertainty "+column, "Reliability "+column, "Temporal correlation "+column, "Geographical correlation "+column, "Product correlation "+column, "Final uncertainty "+column]],on=column, how="inner")
            
        if "data" in column:
            methodologies[column] = methodologies[column].astype('str')
            fit = base_uncertainty_data.rename({"Data":column, "Default base uncertainty": "Default base uncertainty "+column, "Reliability": "Reliability "+column, "Geographical completeness": "Geographical completeness "+column, "Time completeness": "Time completeness "+column, "Temporal correlation": "Temporal correlation "+column, "Geographical correlation": "Geographical correlation "+column, "Product correlation":"Product correlation "+column }, axis =1)
            for index, item in fit.iterrows():
                if str(item[column])=="nan":
                    for i in range(0,6):
                        fit.at[index, list_categories_data[i] + " "+ column + " value"] = 0
                    continue
                else:
                    factors = []
                    for i in range(0,6):
                        factors.append(matrix_data[i,int(item[list_categories_data[i]+" "+column])])
                        # print(list_categories_data[i] + " "+ column, matrix_data[i,int(item[list_categories_data[i]+" "+column])])
                        fit.at[index,list_categories_data[i] + " "+ column]= matrix_data[i,int(item[list_categories_data[i]+" "+column])]
                    fit.at[index,"Final uncertainty "+column] = math.sqrt(np.prod([(factors[i]+1)**2 for i in range(0,len(factors))])*(float(item["Default base uncertainty "+column])+1)**2-1)
            fit[column] = fit[column].astype('str')
            methodologies = pd.merge(methodologies,fit[[column, "Default base uncertainty "+column, "Reliability "+column, "Geographical completeness "+column, "Time completeness "+column, "Temporal correlation "+column, "Geographical correlation "+column, "Product correlation "+column, "Final uncertainty "+column]],on=column, how="inner")
        methodologies.drop_duplicates(inplace=True)
   
    methodologies.to_csv("./Methodologies/assumptions_most_likely.csv")

    methodologies["Total uncertainty"] = 1
    for column in methodologies.columns: 
        if "Final" in column:
            for index, item in methodologies.iterrows():
                if str(methodologies.at[index,column])== "nan":
                    continue
                methodologies.at[index,"Total uncertainty"] = methodologies.at[index,"Total uncertainty"] * (methodologies.at[index,column]+1)**2
    methodologies["Total uncertainty"] = np.sqrt(methodologies["Total uncertainty"]-1)

    methodologies = methodologies.round(1)
    if not(sensitivity_analysis):
        methodologies[['Emission source', 'Description', 'Activity data 1', 'Activity data 2',
       'Factor 1', 'Factor 2', 'Factor 3','Final uncertainty Activity data 1','Final uncertainty Activity data 2','Final uncertainty Factor 1','Final uncertainty Factor 2','Final uncertainty Factor 3',"Total uncertainty"]].to_csv("./Methodologies/results_most_likely.csv")
    
    return methodologies[['Emission source', 'Description', 'Activity data 1', 'Activity data 2',
       'Factor 1', 'Factor 2', 'Factor 3','Final uncertainty Activity data 1','Final uncertainty Activity data 2','Final uncertainty Factor 1','Final uncertainty Factor 2','Final uncertainty Factor 3',"Total uncertainty"]]




def calculations_best_case(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors, sensitivity_analysis = False):
    # Compute base coefficients of variation for each available methodology under assumption most best possible data quality
    for column in methodologies.columns:
        if "Factor" in column:
            methodologies[column] = methodologies[column].astype('str')
            fit = base_uncertainty_factors.rename({"Factor":column, "Default base uncertainty": "Default base uncertainty "+column, "Reliability" : "Reliability "+column, "Temporal correlation" : "Temporal correlation "+column, "Geographical correlation": "Geographical correlation "+column, "Product correlation": "Product correlation "+column }, axis =1)
            for index, item in fit.iterrows():
                if str(item[column])=="nan":
                    for i in range(0,4):
                        item[list_categories_factors[i] + " "+ column + " value"] = 0
                    continue
                else:
                    factors = []
                    for i in range(0,4):
                        # print(list_categories_factors[i] + " "+ column, matrix_factor[i,int(item[list_categories_factors[i]+" "+column])])
                        factors.append(matrix_factor[i,0])
                        fit.at[index,list_categories_factors[i] + " "+ column]= factors[-1]
                    fit.at[index,"Final uncertainty "+column] = math.sqrt(np.prod([(factors[i]+1)**2 for i in range(0,len(factors))])*(float(item["Default base uncertainty "+column])+1)**2-1)
            fit[column] = fit[column].astype('str')
            methodologies = pd.merge(methodologies,fit[[column, "Default base uncertainty "+column, "Reliability "+column, "Temporal correlation "+column, "Geographical correlation "+column, "Product correlation "+column, "Final uncertainty "+column]],on=column, how="inner")
            
        if "data" in column:
            methodologies[column] = methodologies[column].astype('str')
            fit = base_uncertainty_data.rename({"Data":column, "Default base uncertainty": "Default base uncertainty "+column, "Reliability": "Reliability "+column, "Geographical completeness": "Geographical completeness "+column, "Time completeness": "Time completeness "+column, "Temporal correlation": "Temporal correlation "+column, "Geographical correlation": "Geographical correlation "+column, "Product correlation":"Product correlation "+column }, axis =1)
            for index, item in fit.iterrows():
                if str(item[column])=="nan":
                    for i in range(0,6):
                        fit.loc[index,list_categories_data[i] + " "+ column + " value"] = 0
                    continue
                else:
                    factors = []
                    for i in range(0,6):
                        factors.append(matrix_data[i,0])
                        # print(list_categories_data[i] + " "+ column, matrix_data[i,int(item[list_categories_data[i]+" "+column])])
                        fit.loc[index,list_categories_data[i] + " "+ column]= matrix_data[i,0]
                    fit.at[index,"Final uncertainty "+column] = math.sqrt(np.prod([(factors[i]+1)**2 for i in range(0,len(factors))])*(float(item["Default base uncertainty "+column])+1)**2-1)
            fit[column] = fit[column].astype('str')
            methodologies = pd.merge(methodologies,fit[[column, "Default base uncertainty "+column, "Reliability "+column, "Geographical completeness "+column, "Time completeness "+column, "Temporal correlation "+column, "Geographical correlation "+column, "Product correlation "+column, "Final uncertainty "+column]],on=column, how="inner")
        methodologies.drop_duplicates(inplace=True)

    methodologies.to_csv("./Methodologies/assumptions_best.csv")

    methodologies["Total uncertainty"] = 1
    for column in methodologies.columns: 
        if "Final" in column:
            for index, item in methodologies.iterrows():
                if str(methodologies.at[index,column])== "nan":
                    continue
                methodologies.at[index,"Total uncertainty"] = methodologies.at[index,"Total uncertainty"] * (methodologies.at[index,column]+1)**2
    methodologies["Total uncertainty"] = np.sqrt(methodologies["Total uncertainty"]-1)

    methodologies = methodologies.round(1)
    if not(sensitivity_analysis):
        methodologies[['Emission source', 'Description', 'Activity data 1', 'Activity data 2',
       'Factor 1', 'Factor 2', 'Factor 3','Final uncertainty Activity data 1','Final uncertainty Activity data 2','Final uncertainty Factor 1','Final uncertainty Factor 2','Final uncertainty Factor 3',"Total uncertainty"]].to_csv("./Methodologies/results_most_likely.csv")
    
    return methodologies[['Emission source', 'Description', 'Activity data 1', 'Activity data 2',
       'Factor 1', 'Factor 2', 'Factor 3','Final uncertainty Activity data 1','Final uncertainty Activity data 2','Final uncertainty Factor 1','Final uncertainty Factor 2','Final uncertainty Factor 3',"Total uncertainty"]]




def calculations_worst_case(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors, sensitivity_analysis = False):
    # Compute base coefficients of variation for each available methodology under assumption most worst possible data quality
    #
    for column in methodologies.columns:
        if "Factor" in column:
            methodologies[column] = methodologies[column].astype('str')
            fit = base_uncertainty_factors.rename({"Factor":column, "Default base uncertainty": "Default base uncertainty "+column, "Reliability" : "Reliability "+column, "Temporal correlation" : "Temporal correlation "+column, "Geographical correlation": "Geographical correlation "+column, "Product correlation": "Product correlation "+column }, axis =1)
            for index, item in fit.iterrows():
                if str(item[column])=="nan":
                    for i in range(0,4):
                        item[list_categories_factors[i] + " "+ column + " value"] = 0
                    continue
                else:
                    factors = []
                    for i in range(0,4):
                        # print(list_categories_factors[i] + " "+ column, matrix_factor[i,int(item[list_categories_factors[i]+" "+column])])
                        if type(matrix_factor[i,4])== str:
                            factors.append(matrix_factor[i,3])
                        else:
                            factors.append(matrix_factor[i,4])
                        fit.at[index,list_categories_factors[i] + " "+ column]= factors[-1]
                    fit.at[index,"Final uncertainty "+column] = math.sqrt(np.prod([(factors[i]+1)**2 for i in range(0,len(factors))])*(float(item["Default base uncertainty "+column])+1)**2-1)
            fit[column] = fit[column].astype('str')
            methodologies = pd.merge(methodologies,fit[[column, "Default base uncertainty "+column, "Reliability "+column, "Temporal correlation "+column, "Geographical correlation "+column, "Product correlation "+column, "Final uncertainty "+column]],on=column, how="inner")
            
        if "data" in column:
            methodologies[column] = methodologies[column].astype('str')
            fit = base_uncertainty_data.rename({"Data":column, "Default base uncertainty": "Default base uncertainty "+column, "Reliability": "Reliability "+column, "Geographical completeness": "Geographical completeness "+column, "Time completeness": "Time completeness "+column, "Temporal correlation": "Temporal correlation "+column, "Geographical correlation": "Geographical correlation "+column, "Product correlation":"Product correlation "+column }, axis =1)
            for index, item in fit.iterrows():
                if str(item[column])=="nan":
                    for i in range(0,6):
                        item[list_categories_data[i] + " "+ column + " value"] = 0
                    continue
                else:
                    factors = []
                    factor = 0
                    for i in range(0,6):
                        if matrix_data[i,4] != np.nan:
                            factor = matrix_data[i,3]
                            factors.append(matrix_data[i,3])
                        else:
                            factor = matrix_data[i,4]
                            factors.append(matrix_data[i,4])
                        fit.loc[index,list_categories_data[i] + " "+ column]= factor
                    fit.at[index,"Final uncertainty "+column] = math.sqrt(np.prod([(factors[i]+1)**2 for i in range(0,len(factors))])*(float(item["Default base uncertainty "+column])+1)**2-1)
            fit[column] = fit[column].astype('str')
            methodologies = pd.merge(methodologies,fit[[column, "Default base uncertainty "+column, "Reliability "+column, "Geographical completeness "+column, "Time completeness "+column, "Temporal correlation "+column, "Geographical correlation "+column, "Product correlation "+column, "Final uncertainty "+column]],on=column, how="inner")
        methodologies.drop_duplicates(inplace=True)

    methodologies.to_csv("./Methodologies/assumptions_worst.csv")

    methodologies["Total uncertainty"] = 1
    for column in methodologies.columns: 
        if "Final" in column:
            for index, item in methodologies.iterrows():
                if str(methodologies.at[index,column])== "nan":
                    continue
                methodologies.at[index,"Total uncertainty"] = methodologies.at[index,"Total uncertainty"] * (methodologies.at[index,column]+1)**2
    methodologies["Total uncertainty"] = np.sqrt(methodologies["Total uncertainty"]-1)

    methodologies = methodologies.round(1)

    if not(sensitivity_analysis):
        methodologies[['Emission source', 'Description', 'Activity data 1', 'Activity data 2',
       'Factor 1', 'Factor 2', 'Factor 3','Final uncertainty Activity data 1','Final uncertainty Activity data 2','Final uncertainty Factor 1','Final uncertainty Factor 2','Final uncertainty Factor 3',"Total uncertainty"]].to_csv("./Methodologies/results_most_likely.csv")
    
    return methodologies[['Emission source', 'Description', 'Activity data 1', 'Activity data 2',
       'Factor 1', 'Factor 2', 'Factor 3','Final uncertainty Activity data 1','Final uncertainty Activity data 2','Final uncertainty Factor 1','Final uncertainty Factor 2','Final uncertainty Factor 3',"Total uncertainty"]]


def calculations_client_case(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors, sensitivity_analysis=False ):
    # Compute base coefficients of variation for each available methodology under assumption most likely quality of data 
    # return the methodogy csv
    for column in methodologies.columns:
        if "Factor" in column:
            methodologies[column] = methodologies[column].astype('str')
            fit = base_uncertainty_factors.rename({"Factor":column, "Default base uncertainty": "Default base uncertainty "+column, "Reliability" : "Reliability "+column, "Temporal correlation" : "Temporal correlation "+column, "Geographical correlation": "Geographical correlation "+column, "Product correlation": "Product correlation "+column }, axis =1)
            for index, item in fit.iterrows():
                if str(item[column])=="nan":
                    for i in range(0,4):
                        fit.at[index,list_categories_factors[i] + " "+ column + " value"] = 0
                    continue
                else:
                    factors = []
                    for i in range(0,4):
                        # print(list_categories_factors[i] + " "+ column, matrix_factor[i,int(item[list_categories_factors[i]+" "+column])])
                        factors.append(matrix_factor[i,int(item[list_categories_factors[i]+" "+column])])
                        fit.at[index,list_categories_factors[i] + " "+ column]= factors[-1]
                    fit.at[index,"Final uncertainty "+column] = math.sqrt(np.prod([(factors[i]+1)**2 for i in range(0,len(factors))])*(float(item["Default base uncertainty "+column])+1)**2-1)
            fit[column] = fit[column].astype('str')
            methodologies = pd.merge(methodologies,fit[[column, "Default base uncertainty "+column, "Reliability "+column, "Temporal correlation "+column, "Geographical correlation "+column, "Product correlation "+column, "Final uncertainty "+column]],on=column, how="inner")
            
        if "data" in column:
            methodologies[column] = methodologies[column].astype('str')
            fit = base_uncertainty_data.rename({"Data":column, "Default base uncertainty": "Default base uncertainty "+column, "Reliability": "Reliability "+column, "Geographical completeness": "Geographical completeness "+column, "Time completeness": "Time completeness "+column, "Temporal correlation": "Temporal correlation "+column, "Geographical correlation": "Geographical correlation "+column, "Product correlation":"Product correlation "+column }, axis =1)          
            for index, item in fit.iterrows():
                if str(item[column])=="nan":
                    for i in range(0,6):
                        fit.at[index, list_categories_data[i] + " "+ column + " value"] = 0
                    continue
                else:
                    factors = []
                    for i in range(0,6):
                        factors.append(matrix_data[i,int(item[list_categories_data[i]+" "+column])])
                        # print(list_categories_data[i] + " "+ column, matrix_data[i,int(item[list_categories_data[i]+" "+column])])
                        fit.at[index,list_categories_data[i] + " "+ column]= matrix_data[i,int(item[list_categories_data[i]+" "+column])]
                    fit.at[index,"Final uncertainty "+column] = math.sqrt(np.prod([(factors[i]+1)**2 for i in range(0,len(factors))])*(float(item["Default base uncertainty "+column])+1)**2-1)
            fit[column] = fit[column].astype('str')
            methodologies = pd.merge(methodologies,fit[[column,"Available", "Default base uncertainty "+column, "Reliability "+column, "Geographical completeness "+column, "Time completeness "+column, "Temporal correlation "+column, "Geographical correlation "+column, "Product correlation "+column, "Final uncertainty "+column, "Available"]],on=column, how="inner")
        methodologies.drop_duplicates(inplace=True)
   
    methodologies.to_csv("./Methodologies/assumptions_client.csv")

    methodologies["Total uncertainty"] = 1
    for column in methodologies.columns: 
        if "Final" in column:
            for index, item in methodologies.iterrows():
                if str(methodologies.at[index,column])== "nan":
                    continue
                methodologies.at[index,"Total uncertainty"] = methodologies.at[index,"Total uncertainty"] * (methodologies.at[index,column]+1)**2
    methodologies["Total uncertainty"] = np.sqrt(methodologies["Total uncertainty"]-1)

    list_available = [column for column in methodologies.columns if 'Available' in column]
    methodologies["Available"] = methodologies[list_available].all(axis="columns")

    methodologies = methodologies.round(1)
    if not(sensitivity_analysis):
        methodologies[['Emission source', 'Description', 'Activity data 1', 'Activity data 2',
       'Factor 1', 'Factor 2', 'Factor 3','Final uncertainty Activity data 1','Final uncertainty Activity data 2','Final uncertainty Factor 1','Final uncertainty Factor 2','Final uncertainty Factor 3',"Total uncertainty", "Available"]].to_csv("./Methodologies/results_client.csv")
    
    return methodologies[['Emission source', 'Description', 'Activity data 1', 'Activity data 2',
       'Factor 1', 'Factor 2', 'Factor 3','Final uncertainty Activity data 1','Final uncertainty Activity data 2','Final uncertainty Factor 1','Final uncertainty Factor 2','Final uncertainty Factor 3',"Total uncertainty"]]




def main(default=True):
    # Reads model parameters and performs calculations under the most likely, best and worst quality of data assumptions
    directory = ("Methodologies")
    folder_exist = os.path.isdir(directory)
    if not folder_exist:
        os.makedirs("Methodologies")
    methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors = import_model_parameters(default=default)
    if default:
        calculations_underlognormal_assumption(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors)
        calculations_worst_case(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors)
        calculations_best_case(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors)
    if not default:
        calculations_client_case(methodologies, base_uncertainty_factors, base_uncertainty_data, matrix_data, matrix_factor, description_matrix_data, description_matrix_factor, list_categories_data, list_categories_factors)
    print("Done")

# main(False)

