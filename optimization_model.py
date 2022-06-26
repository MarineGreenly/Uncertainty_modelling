from operator import index
from cv2 import merge
from pandas import DataFrame
from pulp import *
from Monte_Carlo_Sum import dict_ADEME, monte_carlo_custom_methodology
from compute_base_CV import *
import os
import matplotlib.pyplot as plt



results_client =  pd.read_csv("Methodologies/results_client.csv")
# computed coefficients of variation for each methodology given the client's data quality input. Additionnally, a column determining
# determining if method can be applied with current data availability

input_client = pd.read_csv("Entry data/Entry data client/review_methodology - client input.csv")
# current results : for each emission source, current used methodology, value per collaborator, pertinence


def handle_data(results_client, input_client):
    # Inputs : results_clients - computed coefficients of variation for each methodology given the client's data quality input
    # current results - for each emission source, current used methodology, value per collaborator, pertinence
    # Outputs : best_available_method - for each scope, coefficients of variation for best available methodology 
    # current_method, all_available_method, pertinent_scopes_non_available

    # Creates output folder to store final data
    directory = ("Output data")
    folder_exist = os.path.isdir(directory)
    if not folder_exist:
        os.makedirs("Output data")
    main(default=False)

    results_client.rename(columns={"Description":"methodology", "Emission source": "scope"}, inplace=True)
    input_client["Current"] = True

    # joins results client and input client on emission source and methodoogy
    merged_method = results_client.join(input_client.set_index(["methodology", "scope"]), on=["methodology","scope"])
    merged_method.drop(["value", "pertinent"], axis=1, inplace=True)
    merged_method = merged_method.join(input_client[["value", "pertinent", "scope"]].set_index("scope"), on="scope")
    # drops non pertinent emission sources
    merged_method = merged_method.mask(~merged_method["pertinent"])
    merged_method.dropna(axis=0, inplace=True, how="all")
    merged_method["Available"].fillna(False, inplace=True)


    # compute methodologies variances
    merged_method["variance"] = merged_method["Total uncertainty"]*merged_method["value"]
    # identifies emissions for each no methodology is available
    merged_method["modelisation disponible"] = True
    merged_method.loc[(merged_method["Total uncertainty"]==0), "modelisation disponible"] = False
    merged_method["Current"].fillna(False, inplace=True)
    merged_method["modelisation disponible"].fillna(False, inplace=True)
    # identifies emissions that clients declared pertinent but for which no modules are available
    pertinent_scopes_non_available = merged_method.mask(merged_method[["Available", "modelisation disponible"]].all(axis=1))

    # groups all information for all available methods
    merged_method = merged_method.mask(~merged_method["Available"])
    merged_method.dropna(axis=0, inplace=True, how="all")
    merged_method = merged_method.mask(~merged_method["modelisation disponible"].astype(bool))
    merged_method.dropna(axis=0, inplace=True, how="all")
    merged_method.reindex()
    all_available_method = merged_method

    # selects information for best, worst available methods, and current method
    best_available_method = merged_method.loc[merged_method.groupby('scope')["variance"].idxmin()]
    worst_available_method = merged_method.loc[merged_method.groupby('scope')["variance"].idxmax()]
    current_method = merged_method.mask(~merged_method["Current"].astype('bool'))

    # fetches default variances of ADEME data to fill in uncomputed variances in current methods
    ademe_default_list = []
    for scope in all_available_method['scope'].unique():
        if scope not in current_method['scope'].unique():
            # sets variance of uncomputed items to the maximum of the computed ADEME variance of the one of the worst available method.
            variance = max(dict_ADEME[int(scope)][1]*float(dict_ADEME[int(scope)][0]),float(worst_available_method.loc[worst_available_method["scope"]==scope, "variance"]))
            ademe_default_list.append({"scope":scope, "variance":variance, "default":True})
    dataframe_to_add = pd.DataFrame(ademe_default_list)
    # adds default values to current_method variances
    current_method.dropna(axis=0, inplace=True, how="all")
    current_method["default"] = False
    current_method = pd.concat([current_method, dataframe_to_add])
    
    # data cleanup after mask operation
    best_available_method.dropna(axis=0, inplace=True, how="all")
    pertinent_scopes_non_available.dropna(axis=0, inplace=True, how="all")
    
    # results storage
    best_available_method.to_csv("Output data/best_available_data_client.csv")
    current_method.to_csv("Output data/current_data_client.csv")
    pertinent_scopes_non_available.to_csv("Output data/pertinent_scopes_non_available_client.csv")
    all_available_method.to_csv("Output data/all_available_methods.csv")

    return best_available_method, current_method, all_available_method, pertinent_scopes_non_available

# print(handle_data(results_client, input_client))


def methodology_choice_model(results_client, input_client):
    # Inputs : results_clients - computed coefficients of variation for each methodology given the client's data quality input
    # current results - for each emission source, current used methodology, value per collaborator, pertinence
    # Outputs : results_dict - dictionnary with value of optimal variables

    best_available_method, current_method, all_available_method, pertinent_scopes_non_available = handle_data(results_client, input_client)
    decision_variables_methodologies = {}
    parameters_current_methologies = {}
    parameters_current_default_variability = {}

    # initializes problem
    problem = LpProblem("test", LpMinimize)

    # decision variables that indicate whether methodology needs to be changed for emission source
    decision_variables_scopes = LpVariable.dicts(name="decision_variables_scopes", indexs=all_available_method["scope"].unique(), lowBound=0, upBound=1, cat=LpBinary)

    for scope in all_available_method["scope"].unique():
        # decision variables that indicate which new methodology has been chosen for emission source
        decision_variables_methodologies[scope] = LpVariable.dicts(name="chosen_methodology_scope_"+str(scope), indexs= all_available_method.loc[all_available_method["scope"]==scope].index.values, lowBound=0, upBound=1, cat=LpBinary)

        # if one new methodology has been chosen, indicator that  methodology needs to be changed for emission source is set to 1
        problem+=lpSum(decision_variables_methodologies[scope][index] for index in all_available_method.loc[all_available_method["scope"]==scope].index.values) == decision_variables_scopes[scope]

        # at most one new methodology per scope
        problem+=lpSum(decision_variables_methodologies[scope][index] for index in all_available_method.loc[all_available_method["scope"]==scope].index.values) <=1

        # computes parameter to help compute total variance based on decision variables
        parameters_current_methologies[scope] = {}
        if scope in current_method.loc[(current_method["default"]==True), "scope"].to_list():
            parameters_current_default_variability[scope]=(1, current_method.loc[(current_method["scope"]==scope), "variance"])
        else:
            parameters_current_default_variability[scope]= (0,0)

        # identifies current methodologies
        for index_methodology in all_available_method.loc[all_available_method["scope"]==scope].index.values:
            problem += decision_variables_methodologies[scope][index_methodology] <= decision_variables_scopes[scope]
            if index_methodology in current_method.loc[(current_method["scope"]==scope) & (current_method["default"]==False)].index.values:
                parameters_current_methologies[scope][index_methodology] = 1
            else:
                parameters_current_methologies[scope][index_methodology] = 0
            
            # cannot chose current methodology as new methodology
            problem += decision_variables_methodologies[scope][index_methodology] <= 1-parameters_current_methologies[scope][index_methodology]

    # best variance possible for clients
    best_var = best_available_method['variance'].sum()

    # chosen variance, depending on decision variables
    chosen_var = lpSum(item['variance']*(decision_variables_methodologies[item["scope"]][index]+parameters_current_methologies[item["scope"]][index]*(1-decision_variables_scopes[item["scope"]])) for index,item in all_available_method.iterrows()) +lpSum(parameters_current_default_variability[scope][0]*parameters_current_default_variability[scope][1]*(1-decision_variables_scopes[scope]) for scope in all_available_method["scope"].unique())

    # Minimize number of methodologies chosen
    problem+=sum(decision_variables_scopes[key] for key in decision_variables_scopes.keys())

    # Chosen variability must be at most 5% away from best variability
    problem+= chosen_var <= 1.05*best_var, "constraint variability"

    # Sets constraint to use method available for uncomputed emission sources
    # if no methodology available because of client data limitation or unavailable module, raises warning
    for index, item in pertinent_scopes_non_available.iterrows():
        if item["Available"] == True and item["modelisation disponible"] == True :
            problem += decision_variables_scopes[item["scope"]] == 1
            pertinent_scopes_non_available.drop(axis=0,index=index, inplace=True)
        elif item["Available"] == False :
            print("Warning. Data is not available to assess pertinent scope  "+item["scope"])
        elif  item["modelisation disponible"] == False :
            print("Warning. No module is available to assess pertinent scope  "+item["scope"])

    
    problem.solve()

    # fills results dict with optimised constraint and variable values
    results_dict  = {}
    for constraint in problem.constraints:
        results_dict[problem.constraints[constraint].name] = problem.constraints[constraint].value()
    for variable in problem.variables():
        results_dict[variable.name] = variable.value()

    return results_dict 

# print(methodology_choice_model(results_client, input_client))


def assess_ADEME_after_optimization(n, assumption): 
    # Goes through all available GHG assessments in the ADEME database, computes uncertainty metrics based on data quality and 
    # methodology assumptions. Recomputes the same statistics after methodology changes ordered by optimization algorithm.
    # Input :
    # Output : ADEME uncertainty assessments results before and after optimization, stored in the Output folder
    
    GHG_assessments = pd.read_csv("Entry data/GHG Assessments.csv")
    list_id = pd.unique(GHG_assessments["id"])
    list_dict = []
    list_initial_results = []
    list_final_results = []

    for id in list_id:
        dict = {}
        assessment = GHG_assessments.loc[GHG_assessments["id"]==id, ["scope_item_id","total_by_staff"]]
        for index, item in assessment.iterrows():
            dict[item["scope_item_id"]] = item["total_by_staff"]
        list_dict.append(dict) 

    # create entry-specific client input (with corresponding values)
    

    
    for i,dict in enumerate(list_dict):
        print("\n", dict)
        
        try:
            input_client_specific = input_client.copy()
            for scope in dict.keys():
                input_client_specific.loc[input_client_specific["scope"]==str(int(scope)),"value"] = dict[scope]
            
            best_available_method, current_method, all_available_method, pertinent_scopes_non_available = handle_data(results_client, input_client_specific)

            # computes total uncertainty
            dict_initial_results = monte_carlo_custom_methodology(dict,n, list_id[i], results_client, methodology_client = input_client_specific)
            dict_initial_results["id"]= list_id[i]
            list_initial_results.append(dict_initial_results)
            
            optimisation_results = methodology_choice_model(results_client, input_client_specific)

            # changes applied methodology based on optimisation results
            for index, item in all_available_method.iterrows():
                if optimisation_results['chosen_methodology_scope_'+str(item["scope"])+"_"+str(int(item["Unnamed: 0"]))] == 1:
                    input_client_specific.loc[input_client["scope"]==item["scope"], "methodology"] = str(item["methodology"])
                    
            # recomputes total uncertainty
            dict_final_results = monte_carlo_custom_methodology(dict,n, list_id[i], results_client, methodology_client = input_client_specific)
            dict_final_results["id"]= list_id[i]
            list_final_results.append(dict_final_results)

        except:
            print("Assessment failed with entry "+str(list_id[i]))


    # Treatment and storage of results
    ADEME_uncertainty_assessment_initial = pd.DataFrame(list_initial_results)

    for k in range(0,4):
        ADEME_uncertainty_assessment_initial.loc[(ADEME_uncertainty_assessment_initial["Probability "+"E"+str(k+1)+">=E"+str(k+2)]==0)| (ADEME_uncertainty_assessment_initial["Probability "+"E"+str(k+1)+">=E"+str(k+2)]==1), ["Probability "+"E"+str(k+1)+">=E"+str(k+2)]  ] = np.nan
    ADEME_uncertainty_assessment_initial.loc[(ADEME_uncertainty_assessment_initial["Probability good order on most important declared scopes"]<10**(-3))| (ADEME_uncertainty_assessment_initial["Probability good order on most important declared scopes"]==1), ["Probability good order on most important declared scopes"]  ] = np.nan

    ADEME_uncertainty_assessment_initial.to_csv("Output data/ADEME_uncertainty_assessment_initial_"+assumption+".csv")
    ADEME_uncertainty_assessment_initial.describe().to_csv("Output data/statistics_ADEME_uncertainty_assessment_initial_"+assumption+".csv")

    ADEME_uncertainty_assessment_final = pd.DataFrame(list_final_results)

    for k in range(0,4):
        ADEME_uncertainty_assessment_final.loc[(ADEME_uncertainty_assessment_final["Probability "+"E"+str(k+1)+">=E"+str(k+2)]==0)| (ADEME_uncertainty_assessment_final["Probability "+"E"+str(k+1)+">=E"+str(k+2)]==1), ["Probability "+"E"+str(k+1)+">=E"+str(k+2)]  ] = np.nan
    ADEME_uncertainty_assessment_final.loc[(ADEME_uncertainty_assessment_final["Probability good order on most important declared scopes"]<10**(-3))| (ADEME_uncertainty_assessment_final["Probability good order on most important declared scopes"]==1), ["Probability good order on most important declared scopes"]  ] = np.nan

    ADEME_uncertainty_assessment_final.to_csv("Output data/ADEME_uncertainty_assessment_final_"+assumption+".csv")
    ADEME_uncertainty_assessment_final.describe().to_csv("Output data/statistics_ADEME_uncertainty_assessment_final_"+assumption+".csv")

    uncertainty_values = ADEME_uncertainty_assessment_final[["Uncertainty measure"]]
    uncertainty_values["initial"] = ADEME_uncertainty_assessment_initial[["Uncertainty measure"]]

    uncertainty_values.rename({"Uncertainty measure": "Final uncertainty measure", "initial": "Initial uncertainty measure"})

    # Plots histogram of uncertainty metrics
    plt.figure()
    plt.hist(uncertainty_values,label=["Final uncertainty measure","Initial uncertainty measure"], range = (0,1), bins = 100, histtype="stepfilled", alpha=0.5)
    plt.xlabel("Uncertainty metric")
    plt.ylabel("Number of entries")
    plt.legend(["Final uncertainty measure","Initial uncertainty measure"])
    plt.savefig("Histogram of uncertainty metric.png")
    plt.close()
    
assess_ADEME_after_optimization(1000, "best") 