# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:21:04 2024

@author: Beth

need to implenet temporal variation- hormone cycles /
immune responses
spatial
possibly input race and form mildly different models- evidence for different microbiomes
bad bacteria and iron /
"""

import random
import matplotlib.pyplot as plt
import numpy as np


class Microbe:
    def __init__(self, species, location):
        self.species = species
        self.location = location

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1))
        self.location = new_location

    def interact(self, estrogen, progesterone, iron_level,
                 acid):
        hormone_level = (estrogen + progesterone) / 2

        # Initialize abundance for the current species if not present
class Good_Bacteria(Microbe):
    def interact(self, iron_level,
                 good_bacteria_abundance, bad_bacteria_abundance,
                 acid, total_good_bacteria_array, step, glycogen,
                 no_good):
        
        if self.species not in total_good_bacteria_array:
            total_good_bacteria_array[self.species] = []

        # Simulate the production of hydrogen peroxide by good bacteria
        else:
            if acid < 2:
                #pH high due to blood (pH7), maybe do hp < something
                good_bacteria_abundance[self.species].append(0)
            elif glycogen > 5:
                good_bacteria_abundance[self.species].append(2)
                glycogen = glycogen - 2
            elif glycogen == 0:
                pass
            else:
                good_bacteria_abundance[self.species].append(1)
                glycogen = glycogen - 1
                # Calculate hydrogen peroxide level based on the abundance of good bacteria
            no_good = sum(good_bacteria_abundance[self.species])
            #print(type(no_good))
        total_good_bacteria_array[self.species].append(no_good)
            
class Bad_Bacteria(Microbe):
    def interact(self, iron_level, bad_bacteria_abundance,
                 acid, total_bad_bacteria, glycogen, tot_bad):
        # Simulate the response of bad bacteria to hydrogen peroxide and iron
        if self.species not in total_bad_bacteria:
            total_bad_bacteria[self.species] = []
        inhibition_factor = 0.2
        if iron_level > inhibition_factor:
            bad_bacteria_abundance[self.species].append(1)
            glycogen -= 1
            iron_level = iron_level - random.uniform(0.1, 0.35)
        elif iron_level > 0.5:
            bad_bacteria_abundance[self.species].append(3)
            glycogen -=3
            iron_level = iron_level - random.uniform(0.25, 0.52)
        tot_bad = sum(bad_bacteria_abundance[self.species])
        total_bad_bacteria[self.species].append(tot_bad)
        
class Immune_Cells():
    def __init__(self, species, location):
        self.species = species
        self.location = location

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1))
        self.location = new_location
    
    def interact(self, total_immune_cells, immune_cell_increase,
                 total_good_bacteria, total_bad_bacteria, iron_level, estrogen,
                 no_immune):
        if self.species not in total_immune_cells:
            total_immune_cells[self.species] = []
            total_immune_cells[self.species].append(no_immune)
            
            immune_cell_increase[self.species] = []
            #immune_cell_increase[self.species].append(no_immune)
        #if self.species not in total_immune_cells:
            
            #total_immune_cells[self.species].append(40)
            #a = []
            #print('hi')
            
        else:
            #print(immune_cell_abundance)
            # if no bacteria presesnt, there's nothing for the immune cell to kill
            if estrogen > 0.6:
                immune_cell_increase[self.species].append(2)
#                    immune_cell_abundance[self.species][-1])
            else:
                immune_cell_increase[self.species].append(1)
                
            if total_good_bacteria["Good_Bacteria_1"][-1] == 0:
                pass
            #calls to simulate immune cell
            else:
                #print(sum(total_immune_cells[self.species]))
                if total_immune_cells[self.species][-1] > 0:
                    for i in range(total_immune_cells[self.species][-1]):
                        if random.randint(1,2) == 1:
                            #immune_cell_increase[self.species][-1] -= -1 
                            #print(immune_cell_increase)
                            immune_response(iron_level, total_good_bacteria,
                                            total_bad_bacteria)
                            
                            
                no_immune = int(sum(immune_cell_increase[self.species]))
                print(immune_cell_increase)
                #no_immune = int(no_immune)
                #print(immune_cell_increase)
                #print(type(no_immune))
                
        total_immune_cells[self.species].append(no_immune)
        print(total_immune_cells)
        #print(no_immune_cells)
        #immune_cell_increase[self.species][0] -=10
                
num_microbes = 4
simulation_steps = 28
num_cycles = 1
cycle_steps = 28
TRIALS = 1
good_bacteria_proportion_tracker = []
tot_good = 0
tot_bad = 0
no_immune_cells = 10


def read_data():
    """
    Returns the data to be used
    -------
    estrogen_data : Array
        Array of estrogen values.
    progesterone_data : Array
        Array of progesterone values.
    maxE : Float
        Maximum estrogen value.
    maxP : Float
        Maximum progesterone value.
    day_no : Array
        Day hormones relate to.

    """
    data = np.genfromtxt("Tabhormone.csv", delimiter = ",", skip_header = 1)
    estrogen_data = data[:,2]
    day_no = data[:,0] + 17
    progesterone_data = data[:,5]
    maxE = np.max(estrogen_data)
    maxP = np.max(progesterone_data)
    return estrogen_data, progesterone_data, maxE, maxP, day_no

def iron_gen(iron, step):
    """
    Parameters
    ----------
    iron : float
        Previous iron level.
    step : integer
        Day within the cycle.

    Returns
    -------
    iron: float
        Iron adgusted by time in hormone cycle
    """
    if  step < 7:
        iron += random.uniform(0, 0.151)
    elif step == 7:
        iron = iron - 0.4
    
    else:
        iron += random.uniform(-0.05, 0.05)
    iron = max(iron, 0)
    return iron

def glycogen_gen(glycogen, estrogen, progesterone):
    hormone_level = (estrogen + progesterone) / 2
    #glycogen levels depend on hormone levels
    if hormone_level > 0.4:
        glycogen = glycogen + random.uniform(0.7, 1)
    elif hormone_level > 0.2:
        glycogen = glycogen + random.uniform(0.2, 0.5)
    return glycogen

def immune_response(iron_level, total_good_bacteria, total_bad_bacteria):
    #randomly determines which cell type to kill
    type_det = random.randint(1,2)
    if type_det == 1:
        #randomly determines bacteria
        which_det = random.randint(1,4)
        #immune cell kills a bacteria
        if total_good_bacteria[f'Good_Bacteria_{which_det}'][-1] > 0:
            total_good_bacteria[f'Good_Bacteria_{which_det}'][-1] -= 1
        #re-calls function if nothing for bacteria to act on
        #else:
         #   immune_response(iron_level, total_good_bacteria, total_bad_bacteria)
        
    else:
        which_det = random.randint(1,2)
        if total_bad_bacteria[f'Bad_Bacteria_{which_det}'][-1] > 0:
            total_bad_bacteria[f'Bad_Bacteria_{which_det}'][-1] -= 1
        #else:
         #   immune_response(iron_level, total_good_bacteria, total_bad_bacteria)
            
    return total_good_bacteria, total_bad_bacteria, iron_level


def hydrogen_peroxide(hydrogen_peroxide_level, total_good_bacteria,
                      total_bad_bacteria, step):
    if step < 7:
        additional_hydrogen_peroxide = - random.uniform(0.5,1)
    else:
        if total_good_bacteria*0.4 > total_bad_bacteria:
            additional_hydrogen_peroxide = random.uniform(0.3,0.6)
        else:
            additional_hydrogen_peroxide = random.uniform(0.1,0.4)   
    hydrogen_peroxide_level += additional_hydrogen_peroxide 
    hydrogen_peroxide_level = max(hydrogen_peroxide_level, 0)
    
    return hydrogen_peroxide_level
        
def simulate_menstrual_cycle(num_cycles, cycle_steps, estrogen, progesterone, maxE, maxP):
    for trial in range(TRIALS):
        estrogen_levels = []
        progesterone_levels = []
        good_bacteria_abundance = {}
        bad_bacteria_abundance = {}
        immune_cell_abundance = {}
        total_bad = {}
        total_good = {}
        total_immune_cells = {}
        total_glycogen = []
        glycogen = 0
        iron_levels = []
        #good_bacteria_total = 0
        #bad_bacteria_total = 0
        good_bacteria_array = []
        bad_bacteria_array = []
        
        
        # Calculate the initial hydrogen peroxide level based on the initial abundance of good bacteria
        initial_hydrogen_peroxide_level = sum(good_bacteria_abundance
                                              [good_species][-1] for good_species
                                              in good_bacteria_abundance)
        hydrogen_peroxide_levels = [initial_hydrogen_peroxide_level]
        # Initialize abundance for each good bacteria species
        for i in range(1, num_microbes + 1):
            good_bacteria_abundance[f"Good_Bacteria_{i}"] = [0]
        for i in range(1, 3):
            bad_bacteria_abundance[f"Bad_Bacteria_{i}"] = [0]
        for cycle in range(num_cycles):
            iron_level = random.uniform(0.5, 0.9)
            #running_good_total = []
            #running_bad_total = []
            for step in range(cycle_steps):
                estrogen_level = estrogen[step]/maxE
                progesterone_level = progesterone[step]/maxP
                estrogen_levels.append(estrogen_level)
                progesterone_levels.append(progesterone_level)
                
                iron_level = iron_gen(iron_level, step)   
                glycogen = glycogen_gen(glycogen, estrogen_level,
                                        progesterone_level)
                hydrogen_peroxide_level = (hydrogen_peroxide
                                           (hydrogen_peroxide_levels[-1],
                                            sum(good_bacteria_array),
                                            sum(bad_bacteria_array), step))
                        
                for i in range(num_microbes):
                    microbe = Microbe(species=f"Microbe_{i}",
                                      location=(random.uniform(0, 10),
                                                random.uniform(0, 10)))
                    microbe.move()
                    #microbe.interact(estrogen_level, progesterone_level, iron_level, good_bacteria_abundance, bad_bacteria_abundance, hydrogen_peroxide_levels, total_glycogen, glycogen)
                
                #good and bad bacteria interactions with environment
                for i in range(1, num_microbes+1):
                    good_bacteria = Good_Bacteria(species=f"Good_Bacteria_{i}",
                                                  location=(random.uniform(0, 10),
                                                            random.uniform(0, 10)))
                    
                    good_bacteria.interact(iron_level, good_bacteria_abundance,
                                           bad_bacteria_abundance,
                                           hydrogen_peroxide_level,
                                           total_good, step, glycogen,tot_good)
                for i in range(1,3):    
                    bad_bacteria = Bad_Bacteria(species=f"Bad_Bacteria_{i}",
                                                location=(random.uniform(0, 10),
                                                          random.uniform(0, 10)))

                    bad_bacteria.interact(iron_level, bad_bacteria_abundance,
                                          hydrogen_peroxide_levels, total_bad,
                                          glycogen, tot_bad)
                immune = Immune_Cells(species = "Immune", location=
                                      (random.uniform(0, 10), random.uniform(0, 10)))
                immune.interact(total_immune_cells, immune_cell_abundance,
                                total_good, total_bad, iron_level,
                                estrogen_level,no_immune_cells)
                #total_good, total_bad, iron_level = immune_response_cycle(immune_cell_no, iron_level, total_good, total_bad)
                hydrogen_peroxide_levels.append(hydrogen_peroxide_level)
                total_glycogen.append(glycogen)
                iron_levels.append(iron_level)
               
                #good_bacteria_array.append(sum(good_bacteria_abundance[good_species][-1]
                #                               for good_species in good_bacteria_abundance))
                #bad_bacteria_array.append(sum(bad_bacteria_abundance[bad_species][-1]
                #                              for bad_species in bad_bacteria_abundance))
                
    return estrogen_levels, progesterone_levels, iron_levels, total_good, total_bad, hydrogen_peroxide_levels, total_glycogen, good_bacteria_array, bad_bacteria_array, total_immune_cells, immune_cell_abundance

def plotting(estrogen, progesterone, iron, good_bacteria, bad_bacteria,
             hydrogen_peroxide_array, glycogen_level, immune_cells, immune_cell_change):
    plt.figure(figsize=(12, 14))
    plot_no = 6
    
    plt.subplot(plot_no, 1, 1)
    plt.plot(range(1, len(estrogen) + 1), estrogen, label='Estrogen Levels',
             linestyle='--')
    plt.plot(range(1, len(progesterone) + 1), progesterone,
             label='Progesterone Levels', linestyle='--')
    plt.title('Hormone Levels Over Time')
    plt.xlabel('Day')
    plt.ylabel('Hormone Levels')
    plt.xlim(0,days)
    plt.legend()
    
    plt.subplot(plot_no, 1, 2)
    plt.plot(range(1, len(iron) + 1), iron, label='Iron Levels',
             linestyle='--', color='orange')
    plt.title('Iron Levels Over Time')
    plt.xlabel('Day')
    plt.ylabel('Iron Levels')
    plt.xlim(0,days)
    plt.legend()
    
    plt.subplot(plot_no, 1, 3)
    for good_species in total_good:
        plt.plot(range(1, len(good_bacteria[good_species]) + 1),
                 good_bacteria[good_species], label=f'{good_species} Abundance')
    for bad_species in total_bad:
            #print(total_bad[bad_species])
        plt.plot(range(1,len(bad_bacteria[bad_species])+1),
                 bad_bacteria[bad_species], label=f'{bad_species} Abundance')
   
    plt.title('Bacteria Abundance Over Time')
    plt.xlabel('Day')
    plt.ylabel('Abundance')
    plt.xlim(0,days)
    plt.legend()
    
    plt.subplot(plot_no, 1, 4)
    plt.plot(range(1, len(glycogen_level) + 1), glycogen_level,
             label = "Glycogen Levels")
    plt.title('Glycogen Abundance Over Time')
    plt.xlabel('Day')
    plt.ylabel('Abundance')
    plt.xlim(0,days)
    plt.legend()
    
    plt.subplot(plot_no, 1, 5)
    plt.plot(range(1, len(hydrogen_peroxide_array) + 1), hydrogen_peroxide_array,
             label='Hydrogen Peroxide Levels', linestyle='--', color='green')
    plt.title('Hydrogen Peroxide Levels')
    plt.xlabel('Day')
    plt.ylabel('Hydrogen Peroxide Levels')
    plt.xlim(0,days)
    plt.legend(loc = "best")
    
    plt.subplot(plot_no, 1, 6)
    plt.plot(range(1, len(immune_cells['Immune']) + 1),
             immune_cells['Immune'], label='Immune Cells',
             linestyle='--', color='green')
    plt.title('Total Immune Cells')
    plt.xlabel('Day')
    plt.ylabel('Abundance')
    plt.xlim(0,days)
    plt.legend(loc = "best")
    '''
    plt.subplot(plot_no, 1, 7)
    plt.plot(range(1, len(immune_cell_change['Immune']) + 1),
             immune_cell_change['Immune'], label='Immune Cells',
             linestyle='--', color='green')
    plt.title('Total Immune Cells')
    plt.xlabel('Day')
    plt.ylabel('Abundance')
    plt.xlim(0,days)
    plt.legend(loc = "best")
    '''
    plt.tight_layout()
    plt.show()
    print("Execution completed.")

est, prog, maxEst, maxProg, day = read_data()
estrogen_levels, progesterone_levels, iron_levels, total_good, total_bad,hydrogen_peroxide_levels, total_glycogen, good_bacteria_array, bad_bacteria_array, total_immune_cells, immune_cell_increase  = simulate_menstrual_cycle(num_cycles, cycle_steps, est, prog, maxEst, maxProg)
days = np.max(day)*num_cycles
plotting(estrogen_levels, progesterone_levels, iron_levels, total_good, total_bad, hydrogen_peroxide_levels, total_glycogen, total_immune_cells, immune_cell_increase)

