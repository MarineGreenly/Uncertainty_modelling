
from re import I
from unittest import result
from bs4 import ResultSet
import matplotlib
import pandas as pd
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import statistics
from itertools import combinations
import gc
from random import randint
import os


results_average = pd.read_csv("Methodologies/results_most_likely.csv")
results_best = pd.read_csv("Methodologies/results_best.csv")
results_worst = pd.read_csv("Methodologies/results_worst.csv")


# create ADEME average dictionnary
statistics_ADEME = pd.read_csv("Entry data/Statistics by scope item.csv")
dict_ADEME = {}
for k in range (1,24):
    if k == 21:
        dict_ADEME[k]= (0,0)
    else:
        dict_ADEME[k] = (statistics_ADEME.loc[(statistics_ADEME['Unnamed: 0'] == str(k)), "total_by_staff.1"].values[0],float(statistics_ADEME.loc[(statistics_ADEME['Unnamed: 0'] == str(k)), "total_by_staff.2"].values[0])/float(statistics_ADEME.loc[(statistics_ADEME['Unnamed: 0'] == str(k)), "total_by_staff.1"].values[0]))

# print(dict_ADEME)
 #no emissions reported on emission source 21


# dic_source_8 = {1 : 82,
# 2 : 82.6,
# 4 : 69.8,
# 6 : 75.39466667}


def monte_carlo(dict,n,assumption, methodology, id = 0, sensitivity_analysis = False, results_var = results_average, random = False):
    #Performs a Monte Carlo simulation based on a dictionnary of means for each ADEME scope,
    #assumption refers to the assumed quality of data and is among ["worst", "best", "average"]
    #methodology refers to the chosen methodology for each ADEME scope and is amonst among ["worst", "best", "average"]
    #if sensitivity analysis is True, results_var, results on methodologies based on tweaked initial variables will be used

    dict_results = {}
    list_results = [0 for i in range (0,n)]
    dict_var = {}
    test_sum = 0
    list_keys=[]
    list_zeros_columns=[]

    ### Fetch methodology coefficient of variations depending on assumptions on quality of data
    if assumption == "best":
        results = results_best
    elif assumption == "worst":
        results = results_worst
    else:
        results = results_average
    if sensitivity_analysis:
        results = results_var


    for key in dict_ADEME.keys():
        # print("Key:", key)

        # Fetch uncertainty for methodology depending on methodology assumption
        str_key = str(int(key))
        if str_key == str(9):
            str_key = "9a"
        if methodology == "best":
            s = np.sqrt(np.log(results.loc[(results["Emission source"] == str_key), "Total uncertainty"].min()**2+1))
            dict_results["Total uncertainty "+str_key] = results.loc[(results["Emission source"] == str_key), "Total uncertainty"].min()
        elif methodology == "worst":
            s = np.sqrt(np.log(results.loc[(results["Emission source"] == str_key), "Total uncertainty"].max()**2+1))
            dict_results["Total uncertainty "+str_key] = results.loc[(results["Emission source"] == str_key), "Total uncertainty"].max()
        else:
            s = np.sqrt(np.log(results.loc[(results["Emission source"] == str_key), "Total uncertainty"].mean()**2+1))
            dict_results["Total uncertainty "+str_key] = results.loc[(results["Emission source"] == str_key), "Total uncertainty"].mean()

        if random:
            possible_methodologies = results.loc[(results["Emission source"] == str_key), "Total uncertainty"]
            i = randint(0,len(possible_methodologies)-1)
            s = np.sqrt(np.log(possible_methodologies.values[i]**2+1))
            dict_results["Total uncertainty "+str_key] = possible_methodologies.values[i]
        
        dict_results["standard deviation "+str_key] = s
        
        
        if not(pd.isna(s)) and s!=0:
            if key not in dict.keys():
                mean = dict_ADEME[key][0]
            else:
                mean = dict[key][0]
                list_keys.append(key)
            if float(mean) == 0:
                list_zeros_columns.append(key)
            test_sum+=float(mean)

            dict_results["mean "+str_key] = mean

            dict_var[key]= stats.lognorm.rvs(s, scale=float(mean), size = n)
            list_results = np.add(list_results, dict_var[key])

    monte_carlo_results = pd.DataFrame.from_dict(dict_var)
    
    # Histogram of most total emissions
    # plt.hist(list_results, bins=200,range=(0,5000),histtype="stepfilled", alpha=1)
    # plt.xlabel("Total Emissions (tCO2e/collaborators)")
    # plt.ylabel("Number of entries")
    # plt.title("Histogram of simulated emissions per collaborators of industrial groups \n based on mean emissions for each emission source")
    # plt.show()
    # plt.savefig("ADEME_uncertainty_assessement/Simulated total per collaborator - ADEME data.png")

    results_dataframe = pd.DataFrame(list_results)
    results_dataframe[0] = results_dataframe[0]/5
    results_dataframe[0] = results_dataframe[0].round()
    results_dataframe[0] = results_dataframe[0]*5

    # print("Most probable value (tCO2e/collaborator)",results_dataframe.mode().loc[0,0])
    dict_results["Most probable total value (tCO2e/collaborator)"] = results_dataframe.median()
    
    # print("Value of sum of averages", test_sum)
    dict_results["Value of sum of declaration and averages"] = test_sum

    ### Computation of probability of right order :
    most_important_scopes = np.argsort(np.array(monte_carlo_results[list_keys].median()))[-5:][::-1] 
    columns = monte_carlo_results.columns
    dict_results["Most important declared scopes"] = columns[most_important_scopes].tolist()

    list_hist = [i for i in columns[most_important_scopes] if i not in list_zeros_columns]
    list_hist.sort()
    data_hist = monte_carlo_results[list_hist]
    
    # Histogram of most important emission sources
    # plt.figure()
    # plt.hist(data_hist,label=list_hist, range = (0,10), bins = 100, histtype="stepfilled", alpha=0.5)
    # plt.xlabel("Total Emissions, tCO2e/collaborator")
    # plt.ylabel("Number of entries")
    # plt.legend(list_hist)
    # plt.savefig("ADEME_uncertainty_assessement/Histogram_most_important_emissions_sources/"+str(id)+assumption+methodology+".png")
    # plt.close()
    
    for i in range(0,4):
        monte_carlo_results["E"+str(i+1)+"<E"+str(i+2)]= np.where(monte_carlo_results[columns[most_important_scopes[i]]]<monte_carlo_results[columns[most_important_scopes[i+1]]], True, False)

    probability = 0
    list = [i for i in range(0,4)]

    for k in range(1,6):
        coef = (-1)**(k+1)
        elements = combinations(list, k)
        for element in elements:
            if k == 1:
                p = len(monte_carlo_results.loc[(monte_carlo_results["E"+str(element[0]+1)+"<E"+str(element[0]+2)])])/n
                # print("Probability "+"E"+str(element[0]+1)+">=E"+str(element[0]+2), 1-p)
                dict_results["Probability "+"E"+str(element[0]+1)+">=E"+str(element[0]+2)] = 1-p
            elif k == 2 :
                p = len(monte_carlo_results.loc[monte_carlo_results["E"+str(element[0]+1)+"<E"+str(element[0]+2)] & monte_carlo_results["E"+str(element[1]+1)+"<E"+str(element[1]+2)]])/n
            elif k == 3 :
                p = len(monte_carlo_results.loc[monte_carlo_results["E"+str(element[0]+1)+"<E"+str(element[0]+2)] & monte_carlo_results["E"+str(element[1]+1)+"<E"+str(element[1]+2)] & monte_carlo_results["E"+str(element[2]+1)+"<E"+str(element[2]+2)] ])/n
            elif k == 4 :
                p = len(monte_carlo_results.loc[monte_carlo_results["E"+str(element[0]+1)+"<E"+str(element[0]+2)] & monte_carlo_results["E"+str(element[1]+1)+"<E"+str(element[1]+2)] & monte_carlo_results["E"+str(element[2]+1)+"<E"+str(element[2]+2)] & monte_carlo_results["E"+str(element[3]+1)+"<E"+str(element[3]+2)]])/n
            elif k == 5 :
                p = len(monte_carlo_results.loc[monte_carlo_results["E"+str(element[0]+1)+"<E"+str(element[0]+2)] & monte_carlo_results["E"+str(element[1]+1)+"<E"+str(element[1]+2)] & monte_carlo_results["E"+str(element[2]+1)+"<E"+str(element[2]+2)] & monte_carlo_results["E"+str(element[3]+1)+"<E"+str(element[3]+2)] & monte_carlo_results["E"+str(element[4]+1)+"<E"+str(element[4]+2)]])/n
            probability = probability + coef*p

    # print("Probability good order over 5 most important emission sources : ", 1 - probability)
    dict_results["Probability good order on most important declared scopes"] = 1 - probability

    if methodology != "best" or assumption != "best":
        best_best = pd.read_csv("ADEME_uncertainty_assessement/ADEME_uncertainty_assessment_best_best.csv")
        reference = best_best.loc[best_best["id"]==id,"Probability good order on most important declared scopes"]
        if float(reference) != 0 :
            if (1 - probability)/float(reference)<1 and (1 - probability)/float(reference)>0:
                dict_results["Uncertainty measure"] = (1 - probability)/float(reference)
            else:
                dict_results["Uncertainty measure"] = np.nan
        else:
            print("Wait ! "+str(id))
    else:
        dict_results["Uncertainty measure"] = 1

    statistics = results_dataframe.describe()
    dict_results["Interval"] = (statistics.loc["25%",0],statistics.loc["75%",0])
    dict_results["Interval lenght"] = statistics.loc["75%",0]-statistics.loc["25%",0]

    print(dict_results)

    del monte_carlo_results
    del results_dataframe
    del dict_var
    gc.collect()

    return dict_results

    

def assess_ADEME(n, assumption, methodology, sensitivity_analysis = False ,results_var = results_average, var=0, random = False): 
    # Goes through all available GHG assessments in the ADEME database, computes uncertainty metrics based on data quality and 
    # methodology assumptions. If sensitivity analysis is True, only computes metrics for last 50 assessments
    # print(results_var)
    GHG_assessments = pd.read_csv("Entry data/GHG Assessments.csv")
    list_id = pd.unique(GHG_assessments["id"])
    list_dict = []
    list_results = []
    for id in list_id:
        dict = {}
        assessment = GHG_assessments.loc[GHG_assessments["id"]==id, ["scope_item_id","total_by_staff"]]
        for index, item in assessment.iterrows():
            dict[item["scope_item_id"]] = item["total_by_staff"]
        list_dict.append(dict) 
    if sensitivity_analysis:
        list_dict = list_dict[-50:-1]       
    for i,dict in enumerate(list_dict):
        # print(dict)
        dict_results = monte_carlo(dict, n, assumption, methodology, id=list_id[i], sensitivity_analysis = sensitivity_analysis, results_var = results_var, random=random)
        dict_results["id"]= list_id[i]
        list_results.append(dict_results)

    ADEME_uncertainty_assessment = pd.DataFrame(list_results)

    for k in range(0,4):
        ADEME_uncertainty_assessment.loc[(ADEME_uncertainty_assessment["Probability "+"E"+str(k+1)+">=E"+str(k+2)]==0)| (ADEME_uncertainty_assessment["Probability "+"E"+str(k+1)+">=E"+str(k+2)]==1), ["Probability "+"E"+str(k+1)+">=E"+str(k+2)]  ] = np.nan
    ADEME_uncertainty_assessment.loc[(ADEME_uncertainty_assessment["Probability good order on most important declared scopes"]<10**(-3))| (ADEME_uncertainty_assessment["Probability good order on most important declared scopes"]==1), ["Probability good order on most important declared scopes"]  ] = np.nan

    if not sensitivity_analysis and not random:
        ADEME_uncertainty_assessment.to_csv("ADEME_uncertainty_assessement/ADEME_uncertainty_assessment_"+assumption+"_"+methodology+".csv")
        ADEME_uncertainty_assessment.describe().to_csv("ADEME_uncertainty_assessement/statistics_ADEME_uncertainty_assessment_"+assumption+"_"+methodology+".csv")

    if random:
        ADEME_uncertainty_assessment.to_csv("ADEME_uncertainty_assessement/ADEME_uncertainty_assessment_"+assumption+"_random"+".csv")
    return ADEME_uncertainty_assessment


def main(random=False):
    directory = ("ADEME_uncertainty_assessement")
    folder_exist = os.path.isdir(directory)
    if not folder_exist:
        os.makedirs("ADEME_uncertainty_assessement")

    list_cases = ["best", "worst", "average"]
    for assumption in list_cases:
        if random:
            assess_ADEME(10000, assumption, "best", random=random)
            print("Done "+assumption+" random")
            continue
        for methodology in list_cases:
            assess_ADEME(10000, assumption, methodology, random=random)
            print("Done "+assumption+"_"+methodology)

# main()



def monte_carlo_custom_methodology(dict,n, id, results_client, methodology_client):
    #Performs a Monte Carlo simulation based on a dictionnary of means for each ADEME scope,
    #assumption refers to the assumed quality of data and is among ["worst", "best", "average"]
    #methodology refers to the chosen methodology for each ADEME scope and is amonst among ["worst", "best", "average"]
    #if sensitivity analysis is True, results_var, results on methdologies based on tweaked initial variables will be used

    dict_results = {}
    list_results = [0 for i in range (0,n)]
    dict_var = {}
    test_sum = 0
    list_keys=[]
    list_zeros_columns=[]

    ### Fetch methodology coefficient of variations depending on assumptions on quality of data
    results = results_client

    for key in dict_ADEME.keys():
        # print("Key:", key)

        # Fetch uncertainty for methodology depending on methodology assumption
        str_key = str(int(key))
        if str_key == str(9):
            str_key = "9a"

        s = np.sqrt(np.log(results.loc[(results["scope"] == str_key) & (results["methodology"] == methodology_client.loc[methodology_client["scope"]==str_key, "methodology"].item()), "Total uncertainty"].mean()**2+1))

        dict_results["Total uncertainty "+str_key] = results.loc[(results["scope"] == str_key), "Total uncertainty"].mean()
        
        dict_results["standard deviation "+str_key] = s
        

        if not(pd.isna(s)) and s!=0:
            if key not in dict.keys():
                mean = dict_ADEME[key][0]
                list_keys.append(key)
            else:
                mean = dict[key]
                list_keys.append(key)
            if float(mean) == 0:
                list_zeros_columns.append(key)
            test_sum+=float(mean)

            dict_results["mean "+str_key] = mean

            dict_var[key]= stats.lognorm.rvs(s, scale=float(mean), size = n)
            list_results = np.add(list_results, dict_var[key])

    monte_carlo_results = pd.DataFrame.from_dict(dict_var)

    results_dataframe = pd.DataFrame(list_results)
    results_dataframe[0] = results_dataframe[0]/5
    results_dataframe[0] = results_dataframe[0].round()
    results_dataframe[0] = results_dataframe[0]*5

    # print("Most probable value (tCO2e/collaborator)",results_dataframe.mode().loc[0,0])
    dict_results["Most probable total value (tCO2e/collaborator)"] = results_dataframe.mode().loc[0,0]
    
    # print("Value of sum of averages", test_sum)
    dict_results["Value of sum of declaration and averages"] = test_sum

    ### Computation of probability of right order :
    most_important_scopes = np.argsort(np.array(monte_carlo_results[list_keys].median()))[-5:][::-1] 
    columns = monte_carlo_results.columns
    dict_results["Most important declared scopes"] = columns[most_important_scopes].tolist()

    list_hist = [i for i in columns[most_important_scopes] if i not in list_zeros_columns]
    list_hist.sort()
    data_hist = monte_carlo_results[list_hist]
    
    for i in range(0,4):
        monte_carlo_results["E"+str(i+1)+"<E"+str(i+2)]= np.where(monte_carlo_results[columns[most_important_scopes[i]]]<monte_carlo_results[columns[most_important_scopes[i+1]]], True, False)

    probability = 0
    list = [i for i in range(0,4)]

    for k in range(1,6):
        coef = (-1)**(k+1)
        elements = combinations(list, k)
        for element in elements:
            if k == 1:
                p = len(monte_carlo_results.loc[(monte_carlo_results["E"+str(element[0]+1)+"<E"+str(element[0]+2)])])/n
                # print("Probability "+"E"+str(element[0]+1)+">=E"+str(element[0]+2), 1-p)
                dict_results["Probability "+"E"+str(element[0]+1)+">=E"+str(element[0]+2)] = 1-p
            elif k == 2 :
                p = len(monte_carlo_results.loc[monte_carlo_results["E"+str(element[0]+1)+"<E"+str(element[0]+2)] & monte_carlo_results["E"+str(element[1]+1)+"<E"+str(element[1]+2)]])/n
            elif k == 3 :
                p = len(monte_carlo_results.loc[monte_carlo_results["E"+str(element[0]+1)+"<E"+str(element[0]+2)] & monte_carlo_results["E"+str(element[1]+1)+"<E"+str(element[1]+2)] & monte_carlo_results["E"+str(element[2]+1)+"<E"+str(element[2]+2)] ])/n
            elif k == 4 :
                p = len(monte_carlo_results.loc[monte_carlo_results["E"+str(element[0]+1)+"<E"+str(element[0]+2)] & monte_carlo_results["E"+str(element[1]+1)+"<E"+str(element[1]+2)] & monte_carlo_results["E"+str(element[2]+1)+"<E"+str(element[2]+2)] & monte_carlo_results["E"+str(element[3]+1)+"<E"+str(element[3]+2)]])/n
            elif k == 5 :
                p = len(monte_carlo_results.loc[monte_carlo_results["E"+str(element[0]+1)+"<E"+str(element[0]+2)] & monte_carlo_results["E"+str(element[1]+1)+"<E"+str(element[1]+2)] & monte_carlo_results["E"+str(element[2]+1)+"<E"+str(element[2]+2)] & monte_carlo_results["E"+str(element[3]+1)+"<E"+str(element[3]+2)] & monte_carlo_results["E"+str(element[4]+1)+"<E"+str(element[4]+2)]])/n
            probability = probability + coef*p

    # print("Probability good order over 5 most important emission sources : ", 1 - probability)
    dict_results["Probability good order on most important declared scopes"] = 1 - probability

    
    best_best = pd.read_csv("ADEME_uncertainty_assessement/ADEME_uncertainty_assessment_best_bestallitems.csv")
    reference = best_best.loc[best_best["id"]==id,"Probability good order on most important declared scopes"]

    if float(reference) != 0 :
        if (1 - probability)/float(reference)<1 and (1 - probability)/float(reference)>0:
            dict_results["Uncertainty measure"] = (1 - probability)/float(reference)
        else:
            dict_results["Uncertainty measure"] = np.nan
    else:
        dict_results["Uncertainty measure"] = 1

    statistics = results_dataframe.describe()
    dict_results["Interval"] = (statistics.loc["25%",0],statistics.loc["75%",0])
    dict_results["Interval lenght"] = statistics.loc["75%",0]-statistics.loc["25%",0]

    del monte_carlo_results
    del results_dataframe
    del dict_var
    gc.collect()

    return dict_results
            