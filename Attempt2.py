# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:21:04 2024

@author: Beth

need to implenet temporal variation- hormone cycles
immune responses
spatial
possibly input race and form mildly different models- evidence for different microbiomes
bad bacteria and iron
"""

import random
import matplotlib.pyplot as plt
import numpy as np
data = np.genfromtxt("Tabhormone.csv", delimiter = ",", skip_header = 1)
maxE = np.max(data[:,2])
maxP = np.max(data[:,5])

class Microbe:
    def __init__(self, species, location):
        self.species = species
        self.location = location

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1), self.location[1] + random.uniform(-1, 1))
        self.location = new_location

    def interact(self, estrogen_level, progesterone_level, iron_level, good_bacteria_abundance, bad_bacteria_abundance, hydrogen_peroxide_levels):
        hormone_level = (estrogen_level + progesterone_level) / 2
        #print(self.species)

        # Initialize abundance for the current species if not present
class Good_Bacteria(Microbe):
    def interact(self, estrogen_level, progesterone_level, iron_level, good_bacteria_abundance, bad_bacteria_abundance, hydrogen_peroxide_levels, total_good, step):
        if self.species not in total_good:#good_bacteria_abundance or total_good:
            #good_bacteria_abundance[self.species] = []
            total_good[self.species] = []

        # Simulate the production of hydrogen peroxide by good bacteria
        else:
            
            if step > 14 and step < 18:
                good_bacteria_abundance[self.species].append(-1)
            else:
                
                good_bacteria_abundance[self.species].append(1)  # Append 1 to indicate the presence of good bacteria for this step
           
                # Calculate hydrogen peroxide level based on the abundance of good bacteria
                #total_good_bacteria = sum(good_bacteria_abundance[good_species][-1] for good_species in good_bacteria_abundance)
        
        
            tot_good = sum(good_bacteria_abundance[self.species])
            total_good[self.species].append(tot_good)
                #print(good_bacteria_abundance)
            hydrogen_peroxide_levels.append(np.abs(tot_good))#* random.randint(1,2))
        
class Bad_Bacteria(Microbe):
    def interact(self, estrogen_level, progesterone_level, iron_level, good_bacteria_abundance, bad_bacteria_abundance, hydrogen_peroxide_levels, total_bad, step):
        # Simulate the response of bad bacteria to hydrogen peroxide
            #print(total_good_bacteria)
        if self.species not in total_bad:
            total_bad[self.species] = []
        inhibition_factor = 0.3 #hydrogen_peroxide_levels[step]/20
        print(iron_level)
        if iron_level > inhibition_factor:
            bad_bacteria_abundance[self.species].append(1)
        elif iron_level > 0.5:
            bad_bacteria_abundance[self.species].append(3)
        tot_bad = np.sum(bad_bacteria_abundance[self.species])
            #print(total_bad)
            # print(tot_bad)
            
        total_bad[self.species].append(tot_bad)
                

num_microbes = 4
simulation_steps = 28
num_cycles = 1
cycle_steps = 28


estrogen_levels = []
progesterone_levels = []
#hydrogen_peroxide_levels = []
good_bacteria_abundance = {}
bad_bacteria_abundance = {}
total_bad = {}
total_good = {}

iron_levels = []
# Initialize abundance for each good bacteria species
for i in range(1, num_microbes + 1):
    good_bacteria_abundance[f"Good_Bacteria_{i}"] = [0]
for i in range(1, 3):
    bad_bacteria_abundance[f"Bad_Bacteria_{i}"] = [0]

# Calculate the initial hydrogen peroxide level based on the initial abundance of good bacteria
initial_hydrogen_peroxide_level = sum(good_bacteria_abundance[good_species][-1] for good_species in good_bacteria_abundance)
hydrogen_peroxide_levels = [initial_hydrogen_peroxide_level]

def simulate_menstrual_cycle(num_cycles, cycle_steps):
    for cycle in range(num_cycles):
        for step in range(cycle_steps):
            estrogen_level = data[step,2]/maxE
            progesterone_level = data[step,5]/maxP
            estrogen_levels.append(estrogen_level)
            progesterone_levels.append(progesterone_level)
            
            if estrogen_level < 0.5 and progesterone_level < 0.5 and step > 14:
                iron_level = random.uniform(0.5, 1.0)
            else:
                iron_level = random.uniform(0, 0.5)
                
            iron_levels.append(iron_level)
            
            for i in range(num_microbes):
                microbe = Microbe(species=f"Microbe_{i}", location=(random.uniform(0, 10), random.uniform(0, 10)))
                microbe.move()
                #microbe.interact(estrogen_level, progesterone_level, iron_level, good_bacteria_abundance, bad_bacteria_abundance, hydrogen_peroxide_levels)
                #if "Good_Bacteria" in microbe.species:
                #    good_bacteria_abundance[microbe.species].append(1)
                #elif "Bad_Bacteria" in microbe.species:
                 #   bad_bacteria_abundance[microbe.species].append(1)
                    
           # for i in range(num_microbes):
                #print(f"Microbe_{i}")
                #microbe = Microbe(species=f"Microbe_{i}", location=(random.uniform(0, 10), random.uniform(0, 10)))
                #microbe.move()
                #microbe.interact(estrogen_level, progesterone_level, iron_level, good_bacteria_abundance, bad_bacteria_abundance, hydrogen_peroxide_levels)
            for i in range(1, num_microbes+1):
                good_bacteria = Good_Bacteria(species=f"Good_Bacteria_{i}", location=(random.uniform(0, 10), random.uniform(0, 10)))
                good_bacteria.interact(estrogen_level, progesterone_level, iron_level, good_bacteria_abundance, bad_bacteria_abundance, hydrogen_peroxide_levels, total_good, step)
            for i in range(1,3):    
                bad_bacteria = Bad_Bacteria(species=f"Bad_Bacteria_{i}", location=(random.uniform(0, 10), random.uniform(0, 10)))
                bad_bacteria.interact(estrogen_level, progesterone_level, iron_level, good_bacteria_abundance, bad_bacteria_abundance, hydrogen_peroxide_levels, total_bad, step)
                
    
'''
            for good_species in good_bacteria_abundance:
                if len(good_bacteria_abundance[good_species]) < num_microbes:
                    good_bacteria_abundance[good_species].append(0)
            for bad_species in bad_bacteria_abundance:
                if len(bad_bacteria_abundance[bad_species]) < num_microbes:
                    bad_bacteria_abundance[bad_species].append(0)       

'''
simulate_menstrual_cycle(num_cycles, cycle_steps)

# Plotting
plt.figure(figsize=(12, 14))

plt.subplot(5, 1, 1)
plt.plot(range(1, len(estrogen_levels) + 1), estrogen_levels, label='Estrogen Levels', linestyle='--')
plt.plot(range(1, len(progesterone_levels) + 1), progesterone_levels, label='Progesterone Levels', linestyle='--')
plt.title('Hormone Levels Over Time')
plt.xlabel('Day')
plt.ylabel('Hormone Levels')
plt.xlim(0,28)
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(range(1, len(iron_levels) + 1), iron_levels, label='Iron Levels', linestyle='--', color='orange')
plt.title('Iron Levels Over Time')
plt.xlabel('Day')
plt.ylabel('Iron Levels')
plt.xlim(0,28)
plt.legend()

plt.subplot(5, 1, 3)
for good_species in total_good:
    plt.plot(range(1, len(total_good[good_species]) + 1), total_good[good_species], label=f'{good_species} Abundance')
plt.title('Good Bacteria Abundance Over Time')
plt.xlabel('Day')
plt.ylabel('Abundance')
plt.xlim(0,28)
plt.legend()

plt.subplot(5, 1, 4)
for bad_species in total_bad:
    #print(total_bad[bad_species])
    plt.plot(range(1,len(total_bad[bad_species])+1), total_bad[bad_species], label=f'{bad_species} Abundance')
plt.title('Bad Bacteria Abundance Over Time')
plt.xlabel('Day')
plt.ylabel('Abundance')
plt.xlim(0,28)
plt.legend()

plt.subplot(5, 1, 5)
plt.plot(range(1, len(hydrogen_peroxide_levels) + 1), hydrogen_peroxide_levels, label='Hydrogen Peroxide Levels', linestyle='--', color='green')
plt.title('Hydrogen Peroxide Levels Produced by Good Bacteria Over Time')
plt.xlabel('Day')
plt.ylabel('Hydrogen Peroxide Levels')
plt.xlim(0,28)
plt.legend()

plt.tight_layout()
plt.show()
print("Execution completed.")