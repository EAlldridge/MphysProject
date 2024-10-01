#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:18:08 2024

@author: matthewillsley
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:59:42 2024

@author: illsl
"""

import random
import matplotlib.pyplot as plt
import numpy as np


NUM_MICROBES = 10
DAYS = 28
NUM_CYCLES = 5000
TRIALS = 1


class Microbe:
    def __init__(self, species, location):
        self.species = species
        self.location = location

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1), self.location[1] + random.uniform(-1, 1))
        self.location = new_location

    def interact(self, estrogen_level, progesterone_level, hydrogen_peroxide_level,iron_level, good_bacteria_count, bad_bacteria_count):
        hormone_level = (estrogen_level + progesterone_level) / 2

        if hormone_level > 0.5 or hydrogen_peroxide_level >= 2*iron_level:

            self.species = random.choice(["Good_Bacteria_1", "Good_Bacteria_2","Good_Bacteria_3", "Good_Bacteria_4"])

        else:
            self.species = random.choice(["Bad_Bacteria_1", "Bad_Bacteria_2"])

#Create specific interactions for the different bacteria

#eg. If good bacteria 1 is produced it secretes h202
class Good_Bacteria_1(Microbe):
    '''
    Class for a specific type of bacteria inside of Microbe. Inherits the same properties
    but has some specific functions eg. h202 secretion
    '''

    def h202_secrete(self,  hydrogen_peroxide_level):
        '''
        Parameters
        ----------
        hydrogen_peroxide_level : Floating point
            The current h2o2 on this step

        Returns
        -------
        Increases the h202 level inside the body making good bacteria more likely

        '''
        hydrogen_peroxide_level += random.uniform(0, 1)

class Bad_Bacteria_1(Microbe):
    pass

    '''def anaerobic_function(self, hydrogen_peroxide_level, total_good_bacteria,total_bad_bacteria,good_bacteria_abundance):

        if total_good_bacteria > total_bad_bacteria:
            good_bacteria_tracker["Good_Bacteria_1"][-1] =- 1
            print("Bacteria 1 Death")
        else:
            hydrogen_peroxide_level -= random.uniform(0, 1)
            print("H202 decreased")'''

def read_data():

    hormone_data = np.genfromtxt("TabHormone.csv" , comments='%',
                       delimiter="," ,skip_header=1,usecols=(0,1,2,3,4,5,6,7))

    day_no = hormone_data[:,0]
    est_level = hormone_data[:,2]
    progest_level = hormone_data[:,5]

    return day_no,est_level,progest_level


def iron_pulse(estrogen_level,progesterone_level):

    if estrogen_level <0.2 and progesterone_level < 0.2: #When hormone levels low, Period occurs
        iron_level = random.uniform(0.7,1)
    else:
        iron_level = random.uniform(0.1, 0.3) #Iron still present but at low levels
    return iron_level

def h202_level(good_bacteria_count,bad_bacteria_count):
    '''

    Parameters
    ----------
    good_bacteria_count : Integer
        DESCRIPTION.
    bad_bacteria_count : Integer
        DESCRIPTION.

    Returns
    -------
    hydrogen_peroxide_level : Float between 0 and 1
        If you have lots of good bacteria they secrete more h202 creating a good
        enviroment, if not you get lower secretions of h2o2

    '''

    if good_bacteria_count > bad_bacteria_count:
        hydrogen_peroxide_level = good_bacteria_count/(good_bacteria_count + bad_bacteria_count)
        hydrogen_peroxide_level=random.uniform(0.7,1)
    else:
        hydrogen_peroxide_level = random.uniform(0.1,0.7)

    return hydrogen_peroxide_level


good_bacteria_proportion_tracker = [] # sets up list to store the final proportion of good bacteria after all cycles


for test in range(TRIALS):

    print("Trial:",test + 1 )

    data = read_data()
    estrogen_levels_raw = data[1]
    estrogen_levels = estrogen_levels_raw/max(estrogen_levels_raw)
    progesterone_levels_raw = data[2]
    progesterone_levels = progesterone_levels_raw/max(progesterone_levels_raw)

    iron_levels = []
    estrogen_levels_over_time = []
    progesterone_levels_over_time = []
    good_bacteria_abundance = {f"Good_Bacteria_{i}": [] for i in range(1, 5)}
    bad_bacteria_abundance = {f"Bad_Bacteria_{i}": [] for i in range(1, 3)}
    total_good_bacteria = []
    total_bad_bacteria = []
    hydrogen_peroxide_levels = []
    good_bacteria_count = 0
    bad_bacteria_count = 0
    hydrogen_peroxide_level = 1


    # New dictionaries to track the sum of bacteria abundances at each step for plotting
    good_bacteria_tracker = {f"Good_Bacteria_{i}": [] for i in range(1, 5)}
    bad_bacteria_tracker = {f"Bad_Bacteria_{i}": [] for i in range(1, 3)}

    for cycle in range(NUM_CYCLES):

        #print(f"\nCycle {cycle + 1} starting...")

        for step in range(DAYS):

            #print(f"Step {step + 1}:")

            # Simulate hormonal changes during the menstrual cycle

            estrogen_level = estrogen_levels[step]
            estrogen_levels_over_time.append(estrogen_level)
            progesterone_level= progesterone_levels[step]
            progesterone_levels_over_time.append(progesterone_level)

            iron_level = iron_pulse(estrogen_level, progesterone_level)
            iron_levels.append(iron_level)

            #print(f"Estrogen Level: {estrogen_level:.2f}")
            #print(f"Progesterone Level: {progesterone_level:.2f}")

            for i in range(NUM_MICROBES):
                microbe = Microbe(species=f"Microbe_{i}", location=(random.uniform(0, 10), random.uniform(0, 10)))
                microbe.move()
                microbe.interact(estrogen_level, progesterone_level, hydrogen_peroxide_level,iron_level, good_bacteria_count, bad_bacteria_count)

                if "Good_Bacteria" in microbe.species:
                    good_bacteria_abundance[microbe.species].append(1)
                    #hydrogen_peroxide_level += random.uniform(0, 1)
                    good_bacteria_count += 1
                elif "Bad_Bacteria" in microbe.species:
                    bad_bacteria_abundance[microbe.species].append(1)
                    #hydrogen_peroxide_level = hydrogen_peroxide_level - random.uniform(0, 1)
                    bad_bacteria_count += 1

                if microbe.species == "Good_Bacteria_1":
                    microbe = Good_Bacteria_1("Good_Bacteria_1",location=(random.uniform(0, 10), random.uniform(0, 10)))
                    #microbe.h202_secrete(hydrogen_peroxide_level)

                elif microbe.species == "Bad_Bacteria_1":
                    microbe  = Bad_Bacteria_1("Bad_Bacteria_1", location=(random.uniform(0, 10), random.uniform(0, 10)))
                    #microbe.anaerobic_function(hydrogen_peroxide_level,total_good_bacteria,total_bad_bacteria, good_bacteria_abundance )

            hydrogen_peroxide_level = h202_level(good_bacteria_count, bad_bacteria_count)

            hydrogen_peroxide_levels.append(hydrogen_peroxide_level)
            total_bad_bacteria.append(bad_bacteria_count)
            total_good_bacteria.append(good_bacteria_count)

            # Ensure each bacteria type's abundance list is padded with zeros if no bacteria appeared in the step
            for good_species in good_bacteria_abundance:
                if len(good_bacteria_abundance[good_species]) < NUM_MICROBES * (cycle + 1):
                    good_bacteria_abundance[good_species].append(0)
            for bad_species in bad_bacteria_abundance:
                if len(bad_bacteria_abundance[bad_species]) < NUM_MICROBES * (cycle + 1):
                    bad_bacteria_abundance[bad_species].append(0)

            # Calculate and store cumulative sums for plotting over time
            for good_species in good_bacteria_abundance:
                current_sum = sum(good_bacteria_abundance[good_species])
                good_bacteria_tracker[good_species].append(current_sum)

            for bad_species in bad_bacteria_abundance:
                current_sum = sum(bad_bacteria_abundance[bad_species])
                bad_bacteria_tracker[bad_species].append(current_sum)

    good_bacteria_proportion = good_bacteria_count/(good_bacteria_count + bad_bacteria_count)
    good_bacteria_proportion_tracker.append(good_bacteria_proportion)

print(f"Average Proportion of good bacteria over, {TRIALS}" ,f"TRIALS is: {np.average(good_bacteria_proportion_tracker):.2f}")

standard_deviation_proportion = np.std(good_bacteria_proportion_tracker)
print(f"The standard deviation of the trials is:{standard_deviation_proportion:.2f}")

# Plotting
plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
plt.plot(range(1, len(estrogen_levels_over_time) + 1), estrogen_levels_over_time , label='Estrogen Levels', linestyle='--')
plt.plot(range(1, len(progesterone_levels_over_time) + 1), progesterone_levels_over_time,  label='Progesterone Levels', linestyle='--')
#plt.plot(range(1, len(iron_levels) + 1), iron_levels , label='Iron Levels', linestyle='--')
#plt.axhline(y=0.2, color='r', linestyle='-', linewidth=1, label='Target Value')
plt.title('Hormone Levels Over Time')
plt.xlabel('Simulation Steps')
plt.ylabel('Hormone Levels')
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(range(1, len(hydrogen_peroxide_levels) + 1), hydrogen_peroxide_levels, label='h2o2 Levels', linestyle='--', color='orange')
plt.title('Hydrogen Peroxide Levels Over time')
plt.xlabel('Simulation Steps')
plt.ylabel('h2o2 Levels')
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(range(1, len(iron_levels) + 1), iron_levels , label='Iron Levels', linestyle='--', color='orange')
plt.title('Iron Pulse')
plt.xlabel('Simulation Steps')
plt.ylabel('Iron Levels')
plt.legend()

plt.subplot(5, 1, 4)
plt.plot(range(1, len(total_bad_bacteria) + 1), total_bad_bacteria, label='Bad bacteria', linestyle='--')
plt.plot(range(1, len(total_good_bacteria) + 1), total_good_bacteria, label='Good bacteria', linestyle='--')
plt.title('Total Microbes')
plt.xlabel('Simulation Steps')
plt.ylabel('# of bacteria')
plt.legend()

plt.subplot(5, 1, 5)
#plt.plot(range(1, len(good_bacteria_proportion_tracker) + 1), good_bacteria_proportion_tracker , label='good bacteria proportion', linestyle='--')
plt.hist(good_bacteria_proportion_tracker,bins= 20,density = False ,label="Distribution of Trials")
plt.title('Histogram of good bacteria for each trial')
plt.xlabel('Proportion of good bacteria')
plt.ylabel('Counts')
plt.legend()




'''plt.subplot(6, 1, 5)
for good_species in good_bacteria_tracker:
    plt.plot(range(1, len(good_bacteria_tracker[good_species]) + 1), good_bacteria_tracker[good_species], label=f'{good_species} Abundance')
plt.title('Good Bacteria Abundance Over Time')
plt.xlabel('Simulation Steps')
plt.ylabel('Abundance')
plt.legend()

plt.subplot(6, 1, 6)
for bad_species in bad_bacteria_tracker:
    plt.plot(range(1, len(bad_bacteria_tracker[bad_species]) + 1), bad_bacteria_tracker[bad_species], label=f'{bad_species} Abundance')
plt.title('Bad Bacteria Abundance Over Time')
plt.xlabel('Simulation Steps')
plt.ylabel('Abundance')
plt.legend()'''

plt.tight_layout()
plt.savefig(fname = "plots", dpi = 100)
plt.show()