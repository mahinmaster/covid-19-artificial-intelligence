#plotting one dependent variable against one independent variable in scatter plot
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd
import numpy as np
from data_accumulation import DataAcc

def scatterPlot(independentVariables, dependent_variable):
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])

    for independentVariable in independentVariables:
        rgb = np.random.rand(3,)
        ax.scatter(dependent_variable, independentVariable, c = rgb)

    ax.set_xlabel('Dependent Variable')
    ax.set_ylabel('Independent Variable')
    ax.set_title('scatter plot')
    plt.show()
    
independent = ['Confirmed', 'Hospitalization_Rate']
dependent = 'Mortality_Rate'
fields = ['Confirmed', 'Hospitalization_Rate','Mortality_Rate','Province_State']

#GET DATA
data_acc = DataAcc()
#print(data_acc)
data_acc.pull_data('4-12-2020','11-08-2020',fields)
#print(data_acc.data)
#len(data_acc.data)
data = data_acc.get_state('Alabama')
data = data.dropna()
print(data)
print(type(data))
data.to_csv('/Users/mahinmaster/Documents/Fall2020/ArtificialIntelligence/CS4100_FinalProject/WeightFinder/output.csv', index = False)
#get list of independent variables
independentVariables = []
for i in independent:
    independentVariables.append(data[i].tolist())
print(independentVariables)

#get list of dependent variables
dependentVariables = data[dependent].tolist()

#print scatterPlot
scatterPlot(independentVariables,dependentVariables)