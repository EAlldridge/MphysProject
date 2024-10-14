# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:21:04 2024

@author: Beth

need to implenet temporal variation- hormone cycles /
immune responses
spatial
possibly input race and form mildly different models- evidence for different microbiomes
bad bacteria and iron /
reintroduce immune cells sequestering iron
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
        return self.location
        print(self.location)
        
    def interact(self, estrogen, progesterone, iron_level,
                 acid):
        hormone_level = (estrogen + progesterone) / 2
    

    
        # Initialize abundance for the current species if not present
class Good_Bacteria(Microbe):
    def interact(self, good_bacteria_abundance, bad_bacteria_abundance,
                 total_good_bacteria_array, step, system,
                 no_good):
 
        if self.species not in total_good_bacteria_array:
            total_good_bacteria_array[self.species] = []

        # Simulate the production of hydrogen peroxide by good bacteria
        else:
            if system["Hydrogen_Peroxide"][step] < 2:
                #pH high due to blood (pH7), maybe do hp < something
                good_bacteria_abundance[self.species].append(0)
            elif system["Glycogen"][step] > 5:
                good_bacteria_abundance[self.species].append(2)
                system["Glycogen"][step] = system["Glycogen"][step] - 2
            elif system["Glycogen"][step] == 0:
                pass
            else:
                good_bacteria_abundance[self.species].append(1)
                system["Glycogen"][step] = system["Glycogen"][step] - 1
                # Calculate hydrogen peroxide level based on the abundance of good bacteria
            no_good = sum(good_bacteria_abundance[self.species])
            #print(type(no_good))
        total_good_bacteria_array[self.species].append(no_good)

class Bad_Bacteria(Microbe):
    def interact(self, bad_bacteria_abundance,
                 total_bad_bacteria, system, tot_bad, step):
        # Simulate the response of bad bacteria to hydrogen peroxide and iron
        if self.species not in total_bad_bacteria:
            total_bad_bacteria[self.species] = []
        inhibition_factor = 0.2
        if system["Iron"][step] > inhibition_factor:
            bad_bacteria_abundance[self.species].append(1)
            #print(system)
            #print(system["Glycogen"])
            system["Glycogen"][-1] -= 1
            system["Iron"][step] = system["Iron"][step] - random.uniform(0.1, 0.35)
        elif system["Iron"][step] > 0.5:
            bad_bacteria_abundance[self.species].append(3)
            system["Glycogen"][-1] -= 3
            system["Iron"][step] = system["Iron"][step] - random.uniform(0.25, 0.52)
        tot_bad = sum(bad_bacteria_abundance[self.species])
        total_bad_bacteria[self.species].append(tot_bad)

class Immune_Cells():
    def __init__(self, species, location):
        self.species = species
        self.location = location
        #self.time = time

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1))
        self.location = new_location
    
    def interact(self, total_immune_cells, immune_cell_increase,
                 total_good_bacteria, total_bad_bacteria, system,
                 no_immune, step):
        if self.species not in total_immune_cells:
            total_immune_cells[self.species] = []
            total_immune_cells["Time"] = []
            total_immune_cells[self.species].append(no_immune)

            immune_cell_increase[self.species] = []
            
        else:
            #print(immune_cell_abundance)
            # if no bacteria presesnt, there's nothing for the immune cell to kill
            if system["Estrogen"][step] > 0.6:
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
                            immune_cell_increase[self.species][-1] -= 1

                            #print(immune_cell_increase)
                            immune_response(system["Iron"][step], total_good_bacteria,
                                            total_bad_bacteria)
                            #print(total_immune_cells)
                            #t = Immune_Cells(species = "Time",location=
                             #                (random.uniform(0, 10), random.uniform(0, 10)))
                            #t.time(i)

                no_immune = int(sum(immune_cell_increase[self.species]))
            total_immune_cells[self.species].append(no_immune)

        def time(self, i):
            if self.time not in total_immune_cells:
                total_immune_cells[self.time] = np.arange(0,0, len(total_immune_cells["Cells"]))
            total_immune_cells[self.time][i] +=1
            if  total_immune_cells[self.time][i] >=2:
                total_immune_cells[self.time][i] -=2

class Environment():
    def __init__(self, factor):
        self.factor = factor
    def horm(self, system, hormone, max_level, step):
        if self.factor not in system:
            system[self.factor] = []
        system[self.factor].append(hormone[step]/ max_level)
        
    def glycogen_c(self, system, step):
        if self.factor not in system:
            system[self.factor] = []
        system[self.factor].append(glycogen_gen(system, step))
        
    def hydrogen_peroxide_fun(self, system, step, total_good_bacteria,
                              total_bad_bacteria):
        if self.factor not in system:
            system[self.factor] = []
        system[self.factor].append(hydrogen_peroxide(system, total_good_bacteria,
                              total_bad_bacteria, step))
    def iron_gen(self, system, step):
        if self.factor not in system:
            system[self.factor] = []
        system[self.factor].append(iron_gen(system, step))
 
num_microbes = 4
simulation_steps = 28
num_cycles = 1
cycle_steps = 28
TRIALS = 1
good_bacteria_proportion_tracker = []
tot_good = 0
tot_bad = 0
no_immune_cells = 50

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

def iron_gen(system, step):
    hormone_level = (system["Estrogen"][step] + system["Progesterone"][step])/2
    if  hormone_level < 0.2:
        iron = random.uniform(0.6, 0.9)
    else:
        iron = random.uniform(0.1, 0.4)
    #else:
     #   iron = random.uniform(0.05, 0.05)

    return iron

def glycogen_gen(system, step):
    hormone_level = (system["Estrogen"][step] + system["Progesterone"][step]) / 2
    #glycogen levels depend on hormone levels
    if len(system["Glycogen"]) == 0:
        gly = random.uniform(2, 4)
    else:
        #print(system["Glycogen"])
        if hormone_level > 0.4:
            gly = system["Glycogen"][-1] + random.uniform(5, 9)
            
        elif hormone_level > 0.2:
            gly = system["Glycogen"][-1] + random.uniform(4, 7)
        else:
            gly = system["Glycogen"][-1] + random.uniform(2, 3)
    #print(gly)
    return gly

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
            
    return total_good_bacteria, total_bad_bacteria

def hydrogen_peroxide(system, total_good_bacteria, total_bad_bacteria, step):
    if len(system["Hydrogen_Peroxide"]) == 0:
        hydrogen_peroxide = total_good_bacteria
        return hydrogen_peroxide
    elif step < 7:
        additional_hydrogen_peroxide = - random.uniform(0.5,1)
    else:
        if total_good_bacteria*0.4 > total_bad_bacteria:
            additional_hydrogen_peroxide = random.uniform(0.3,0.6)
        else:
            additional_hydrogen_peroxide = random.uniform(0.1,0.4) 
    hydrogen_peroxide = max((system["Hydrogen_Peroxide"][-1] + additional_hydrogen_peroxide),0)
            
    return hydrogen_peroxide 
 
def simulate_menstrual_cycle(num_cycles, cycle_steps, estrogen, progesterone, maxE, maxP):
    for trial in range(TRIALS):
        good_bacteria_abundance = {}
        bad_bacteria_abundance = {}
        immune_cell_abundance = {}
        system = {}
        total_bad = {}
        total_good = {}
        total_immune_cells = {}
        good_bacteria_array = []
        bad_bacteria_array = []
 
        # Initialize abundance for each good bacteria species
        for i in range(1, num_microbes + 1):
            good_bacteria_abundance[f"Good_Bacteria_{i}"] = [0]
        for i in range(1, 3):
            bad_bacteria_abundance[f"Bad_Bacteria_{i}"] = [0]
        
        for cycle in range(num_cycles):
            #running_good_total = []
            #running_bad_total = []
            for step in range(cycle_steps):
                
                hormone_comp = Environment(factor = "Estrogen")
                hormone_comp.horm(system, estrogen, maxE, step)
                hormone_comp = Environment(factor = "Progesterone")
                hormone_comp.horm(system, progesterone, maxP, step)
                
                i = Environment(factor = "Iron")
                i.iron_gen(system, step)   
                g = Environment(factor = "Glycogen")
                g.glycogen_c(system, step)
                h = Environment(factor = "Hydrogen_Peroxide")
                h.hydrogen_peroxide_fun(system, step, sum(good_bacteria_array),
                sum(bad_bacteria_array))
                
                #good and bad bacteria interactions with environment
                for i in range(1, num_microbes+1):
                    good_bacteria = Good_Bacteria(species=f"Good_Bacteria_{i}",
                                                  location=(random.uniform(0, 10),
                                                            random.uniform(0, 10)))
                    
                    good_bacteria.interact(good_bacteria_abundance,
                                           bad_bacteria_abundance,
                                           total_good, step, system, tot_good)
                for i in range(1,3):    
                    bad_bacteria = Bad_Bacteria(species=f"Bad_Bacteria_{i}",
                                                location=(random.uniform(0, 10),
                                                          random.uniform(0, 10)))

                    bad_bacteria.interact(bad_bacteria_abundance, 
                                          total_bad, system, tot_bad, step)
                immune = Immune_Cells(species = "Cells", location=
                                      (random.uniform(0, 10), random.uniform(0, 10)))
                immune.interact(total_immune_cells, immune_cell_abundance,
                                total_good, total_bad,
                                system, no_immune_cells, step)
                
    return system, total_good, total_bad, total_immune_cells

def plotting(system, good_bacteria, bad_bacteria, immune_cells):
    plt.figure(figsize=(12, 14))
    plot_no = 6
    
    plt.subplot(plot_no, 1, 1)
    plt.plot(range(1, len(system["Estrogen"]) + 1), system["Estrogen"], label='Estrogen Levels',
             linestyle='--')
    plt.plot(range(1, len(system["Progesterone"]) + 1), system["Progesterone"],
             label='Progesterone Levels', linestyle='--')
    plt.title('Hormone Levels Over Time')
    plt.xlabel('Day')
    plt.ylabel('Hormone Levels')
    plt.xlim(0,days)
    plt.legend()
    
    plt.subplot(plot_no, 1, 2)
    plt.plot(range(1, len(system["Iron"]) + 1), system["Iron"], label='Iron Levels',
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
    plt.plot(range(1, len(system["Glycogen"]) + 1), system["Glycogen"],
             label = "Glycogen Levels")
    plt.title('Glycogen Abundance Over Time')
    plt.xlabel('Day')
    plt.ylabel('Abundance')
    plt.xlim(0,days)
    plt.legend()
    
    plt.subplot(plot_no, 1, 5)
    plt.plot(range(1, len(system["Hydrogen_Peroxide"]) + 1), system["Hydrogen_Peroxide"],
             label='Hydrogen Peroxide Levels', linestyle='--', color='green')
    plt.title('Hydrogen Peroxide Levels')
    plt.xlabel('Day')
    plt.ylabel('Hydrogen Peroxide Levels')
    plt.xlim(0,days)
    plt.legend(loc = "best")
    
    plt.subplot(plot_no, 1, 6)
    plt.plot(range(1, len(immune_cells['Cells']) + 1),
             immune_cells['Cells'], label='Immune Cells',
             linestyle='--', color='green')
    plt.title('Total Immune Cells')
    plt.xlabel('Day')
    plt.ylabel('Abundance')
    plt.xlim(0,days)
    plt.legend(loc = "best")
    '''
    plt.subplot(plot_no, 1, 7)
    
    for good_species in total_good:
        positions = Good_Bacteria.move()
        plt.plot(positions, label='Immune Cells',
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
system, total_good, total_bad, total_immune_cells  = simulate_menstrual_cycle(num_cycles, cycle_steps, est, prog, maxEst, maxProg)
days = np.max(day)*num_cycles
plotting(system, total_good, total_bad, total_immune_cells)
