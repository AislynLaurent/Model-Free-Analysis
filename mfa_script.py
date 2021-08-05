## IMPORT
# Standard
import sys
import os
import re
import csv
import glob
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
try: 
    # import matplotlib
    from matplotlib import pyplot as plt
except ImportError:
    sys.exit('MatPlotLib (https://matplotlib.org/) module not found. Terminating...')

# DECLARE
# Now
now = datetime.now()

# Does a fourier transform on simulated data
# input: array of q values, and a 2D array of simulated values (z, SLD)
# output: result array of summed FT
def sim_fourier(q, sim_values):
    sim_values_tranformed = []
    result = []

    for qvalue in q:
        result.clear()

        for sld, z in zip(sim_values[1], sim_values[0]):
            # We use *10 because gromacs outputs z values as nm. We need angstrom
            sA = sld * (np.cos(qvalue * z * 10) + 1j * np.sin(qvalue * z * 10))
            result.append(sA)

        j = sum(result)
        sim_values_tranformed.append(j)

    return sim_values_tranformed

# Minimize setup
def model(k, sim_i_values, c):
    return ( k * np.array(sim_i_values) + c )

def calculated_model(parameters, sim_i_values):
    k = parameters['k'].value
    c = parameters['c'].value

    return model(k, sim_i_values, c)

def objective_function(parameters, x, sim_i_values, exp_i_values, exp_e_values):
    residual = exp_i_values[min_index:max_index] - calculated_model(parameters, sim_i_values[min_index:max_index])
    weighted_residual = np.power(residual, 2) / np.power(exp_e_values[min_index:max_index], 2)
    return weighted_residual

## HOUSEKEEPING
print('\n------------------------------------------------\n Model Free Analysis Script\t v.0.1\n\n\tWritten By:\t Aislyn Laurent\n\tLast Edited:\t 26-04-2020\n-------------------------------------------------\n')

## USER INPUT
# Data Files
print('Please enter the following information:\n')
data_experimental = input("Experimental data file path: ")
data_simulated = input("Simulated data file path: ")

# Data Range
max_value = float(input("\nMaximum x value to consider: "))
min_value = float(input("Minimum x value to consider: "))

#Background for the experimental data
background = float(input("\nExperimental backgound value: "))

# Ask the user if they want to plot a graph
plot_graph_input = input("\nWould you like to generate a graph at the end of the process? [Y/N]: ")

if plot_graph_input == 'Y' or plot_graph_input == 'y':
    plot_graph = True
else:
    plot_graph = False

if data_experimental and data_simulated and max_value and min_value != None:
    print('\t\t\t\t\tSuccess!')
else:
    print('Some error has occured - terminating...')
    sys.exit('Input error - critical data missing.')

print('\n------------------------------------------------\n')

## PROCESS DATA
values_experimental = []
values_simulated = []
values_experimental_formfactor = []
values_simulated_formfactor = []

# Get data experimental
q_value = []
i_value = []
e_value = []

try: 
    exp_file = open(data_experimental, 'r')
    print('Loading experimental data... ', end =" ")
except OSError:
    sys.exit('Experimental data file path \"'+data_experimental+'\" is invalid. Terminating...')

#loop over the lines and save them. If error, store as string and then display
for line in exp_file:
    line = line.strip()
    
    # check for letters / words / headers / footers
    # I removed .isdigit because it would consider my negative numbers as the header and ignore them
    if not line[:1] or re.search('[a-df-zA-DF-Z]', line):
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

print('\t\tSuccess!')
print('Processing experimental values...', end =" ")

values_experimental.append(q_value)
values_experimental.append(i_value)
values_experimental.append(e_value)
values_experimental = np.array(values_experimental)
print(values_experimental.shape)

# Get data from simulated
z_value = []
sld_value = []

try:
    sim_file = open(data_simulated, 'r')
    print('Loading simulated data... ', end =" ")
except OSError:
    sys.exit('Simulated data file path \"'+data_simulated+'\" is invalid. Terminating...')

#loop over the lines and save them. If error, store as string and then display
for line in sim_file:
    line = line.strip()
    
    # check for letters / words / headers / footers
    # I removed .isdigit because it would consider my negative numbers as the header and ignore them
    if not line[:1] or re.search('[a-df-zA-DF-Z]', line):
        pass
    else:
        fields = line.split()

        z_value.append(float(fields[0]))
        sld_value.append(float(fields[1]))

print('\t\tSuccess!')
print('Processing simulated values...', end =" ")

# not q, i and e. Should be z and SLD, respectively. No third column
values_simulated.append(z_value)
values_simulated.append(sld_value)
values_simulated = np.array(values_simulated)
print(values_simulated.shape, end =" ")

print('\n------------------------------------------------\n')

print('\nSetting value range...')

max_index = min(enumerate(values_experimental[0]), key=lambda x: abs(max_value - x[1]))
min_index = min(enumerate(values_experimental[0]), key=lambda x: abs(min_value - x[1]))

print('Max index: '+str(max_index[0])+'\t\tMax value: '+str(max_index[1]))
print('Min index: '+str(min_index[0])+'\t\tMin value: '+str(min_index[1]))

max_index = max_index[0]
min_index = min_index[0]

print('\n\t\t\t\t\tSuccess!')

print('\n------------------------------------------------\n')

print('FT on SLD of Simulation Data...', end =" ")

#SLD background subtraction
solvent_sld = values_simulated[1][0]
values_simulated[1] = values_simulated[1] - solvent_sld

values_simulated_transformed = sim_fourier(values_experimental[0], values_simulated)

print('\tSuccess!')

print('\n------------------------------------------------\n')

print('Converting FT (i.e. Scattering Amplitude) into Form Factor...', end =" ")

values_simulated_formfactor = [abs(x) for x in values_simulated_transformed]

print('\tSuccess!')

print('\n------------------------------------------------\n')

print('Converting Experimental Iq and Error to Form Factor...', end =" ")

values_experimental_formfactor = values_experimental[0] * np.sign(values_experimental[1]) * np.sqrt(abs((values_experimental[1])-background))
values_experimental_formfactor_error = (values_experimental[2] / values_experimental[1]) * values_experimental_formfactor

print('\tSuccess!')

print('\n------------------------------------------------\n')

# Sum [(ExpI - k * SimI + c)^2/(ExpError)^2] minimize that expression by varying k and c

print('Fitting...', end =" ")

# Parameter setup
parameters = lsq.Parameters()
parameters.add_many(
    ('k', 1, True),
    ('c', 0, True)
)

x = None

fit_result = lsq.minimize(
    objective_function,
    parameters,
    #have to adjust
    args=(x, values_simulated_formfactor, values_experimental_formfactor, values_experimental_formfactor_error)
)

calculated_k = fit_result.params['k'].value
calculated_c = fit_result.params['c'].value
calculated_sim_values = model(calculated_k, values_simulated_formfactor, calculated_c)

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

    writer.writerow(['Experimental FormFactor:'])
    writer.writerow([])

    for exp_ff in values_experimental_formfactor:
        writer.writerow([exp_ff])

    writer.writerow([])
    writer.writerow(['------------------------------------------------'])
    writer.writerow([])

    writer.writerow(['Fit Data:'])
    writer.writerow([])
    writer.writerow(['q','FormFactor'])

    for fit_q_value, fit_i_value in zip(values_experimental[0], calculated_sim_values):
        writer.writerow([fit_q_value, fit_i_value])
    
    writer.writerow([])
    writer.writerow(['------------------------------------------------'])
    writer.writerow([])

print('\n\t\t\t\t\tSuccess!')

print('\n------------------------------------------------\n')

print('Job complete!')

if plot_graph:
    print('Plotting results...')

    plt.figure()

    plt.scatter(
            values_experimental[0][min_index:max_index],
            calculated_sim_values[min_index:max_index],
            edgecolor='r',
            facecolor='w',
            label='Scaled Simulated FormFactor',
            zorder=5
        )

    plt.errorbar(
        values_experimental[0][min_index:max_index],
        values_experimental_formfactor[min_index:max_index],
        fmt='o',
        color='c',
        mfc='w',
        label='Experimental FormFactor',
        zorder=1
    )

    plt.errorbar(
        values_experimental[0][min_index:max_index],
        values_experimental_formfactor[min_index:max_index] - values_experimental_formfactor_error[min_index:max_index],
        fmt='_',
        color='grey',
        zorder=0
    )
    plt.errorbar(
        values_experimental[0][min_index:max_index],
        values_experimental_formfactor[min_index:max_index] + values_experimental_formfactor_error[min_index:max_index],
        fmt='_',
        color='grey',
        zorder=0
    )

    plt.xlabel('q(A-1)')
    plt.ylabel('FormFactor')
    plt.title('Simulated vs Experimental FormFactor')
    plt.legend()

    plt.show()

print('\n------------------------------------------------\n')

print('process terminating...')
print('Thanks for playing!')

print('\n------------------------------------------------\n')