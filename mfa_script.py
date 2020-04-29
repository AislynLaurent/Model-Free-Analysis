## IMPORT
# Standard
import sys
import os
import re
import csv
from datetime import datetime
# Math
try:
    import numpy as np 
except ImportError:
    sys.exit('Numpy (https://numpy.org/) module not found. Terminating...')
try:
    import lmfit as lsq
except ImportError:
    sys.exit('LMFit (https://lmfit.github.io/lmfit-py/) module not found. Terminating...')

## DECLARE
# Now
now = datetime.now()

# Minimize setup
def model(k, sim_i_values, c):
    return (k * sim_i_values + c )

def calculated_model(paramters, sim_i_values):
    k = paramters['k'].value
    c = paramters['c'].value

    return model(k, sim_i_values, c)

def objective_function(paramters, x, sim_values, exp_values):

    residual = exp_values[1][min_index:max_index] - calculated_model(paramters, sim_values[1][min_index:max_index])
    weighted_residual = np.power(residual, 2) / np.power(exp_values[2][min_index:max_index], 2)

    return weighted_residual

## HOUSEKEEPING
print('\n------------------------------------------------\n Model Free Analysis Script\t v.0.1\n\n\tWritten By:\t Aislyn Laurent\n\tLast Edited:\t 26-04-2020\n-------------------------------------------------\n')

## USER INPUT
# Data Files
print('Please enter the following information:')
data_experimental = input('\nExperimental data file path:\n')
data_simulated = input('\nSimulated data file path:\n')

# Data Range
max_value = float(input('\nData max value:\n'))
min_value = float(input('\nData min value:\n'))

if data_experimental and data_simulated and max_value and min_value != None:
    print('\t\t\t\t\tSuccess!')
else:
    print('Some error has occured - terminating...')
    sys.exit('Input error - critical data missing.')

print('\n------------------------------------------------\n')

## PROCESS DATA
values_experimental = []
values_simulated = []

# Get data from files
for i in range(2):
    q_value = []
    i_value = []
    e_value = []

    if i == 0:
        try: 
            file_data = open(data_experimental, 'r')
            print('Loading experimental data... ', end =" ")
        except OSError:
            sys.exit('Experimental data file path \"'+data_experimental+'\" is invalid. Terminating...')
    else:
        try:
            file_data = open(data_simulated, 'r')
            print('Loading simulated data... ', end =" ")
        except OSError:
            sys.exit('Simulated data file path \"'+data_simulated+'\" is invalid. Terminating...')

    print('\t\tSuccess!')
    print('Processing values...', end =" ")

    #loop over the lines and save them. If error, store as string and then display
    for line in file_data:
        line = line.strip()
        
        # check for letters / words / headers / footers
        if not line[:1].isdigit() or re.search('[a-df-zA-DF-Z]', line):
            pass
        else:
            fields = line.split()

            q_value.append(float(fields[0]))
            i_value.append(float(fields[1]))
            try:
                if fields[2] == 0:
                    e_value.append(1)
                else:
                    e_value.append(float(fields[2]))
            except IndexError:
                e_value.append(1)

    if i ==0:
        values_experimental.append(q_value)
        values_experimental.append(i_value)
        values_experimental.append(e_value)
        values_experimental = np.array(values_experimental)
        print(values_experimental.shape, end =" ")
    else:
        values_simulated.append(q_value)
        values_simulated.append(i_value)
        values_simulated.append(e_value)
        values_simulated = np.array(values_simulated)
        print(values_simulated.shape, end =" ")
    
    print('\t\tSuccess!')

print('\n------------------------------------------------\n')

print('Interpolating simulated values...', end =" ")

values_simulated_inter = []
values_simulated_inter.append(values_experimental[0])
values_simulated_inter.append(np.interp(values_experimental[0], values_simulated[0], values_simulated[1]))

print('\tSuccess!')

print('\nSetting value range...')

max_index = min(enumerate(values_experimental[0]), key=lambda x: abs(max_value - x[1]))
min_index = min(enumerate(values_experimental[0]), key=lambda x: abs(min_value - x[1]))

print('Max index: '+str(max_index[0])+'\t\tMax value: '+str(max_index[1]))
print('Min index: '+str(min_index[0])+'\t\tMin value: '+str(min_index[1]))

max_index = max_index[0]
min_index = min_index[0]

print('\n\t\t\t\t\tSuccess!')

print('\n------------------------------------------------\n')

# Sum [(E - k*S+c)^2/(errE)^2] minimize that expression by varying k and c
# Sum [(E - (k*S+c))^2/(errE)^2

print('Fitting...', end =" ")

# Parameter setup
parameters = lsq.Parameters()
parameters.add_many(('k', 1, True), ('c', 0, True))

x = None

fit_result = lsq.minimize(
    objective_function,
    parameters,
    args=(x, values_simulated, values_experimental)
)

print('\t\t\t\tSuccess!')

print('\n------------------------------------------------\n')

fit_stats = lsq.fit_report(fit_result)
print(fit_stats)

print('\n------------------------------------------------\n')

output_filename = 'mfa_result_'+now.strftime("%m-%d-%H%M")+'.csv'

print('Writting results to \"'+output_filename+'\"...')

with open(output_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(['Model Free Analysis Results for '+now.strftime("%m/%d/%H:%M")])
    writer.writerow([])
    writer.writerow(['Experiemental dataset: '+data_experimental])
    writer.writerow(['Simulated dataset: '+data_simulated])

    writer.writerow([])
    writer.writerow(['------------------------------------------------'])
    writer.writerow([])

    writer.writerow(['Fit Statistics'])
    writer.writerow([])
    writer.writerow([fit_stats])

    writer.writerow([])
    writer.writerow(['------------------------------------------------'])
    writer.writerow([])

    writer.writerow(['Fit Data:'])
    writer.writerow([])
    writer.writerow(['q','i'])

    for fit_q_value, fit_i_value in zip(values_experimental[0], fit_result.residual):
        writer.writerow([fit_q_value, fit_i_value])
    
    writer.writerow([])
    writer.writerow(['------------------------------------------------'])
    writer.writerow([])

print('\n\t\t\t\t\tSuccess!')

print('\n------------------------------------------------\n')

print('Job complete, process terminating...')
print('Thanks for playing!')

print('\n------------------------------------------------\n')