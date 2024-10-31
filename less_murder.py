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
only one grow per bacteria?
17/10/24
Started: 11:20
"""

import random
import matplotlib.pyplot as plt
import numpy as np


num_microbes = 4
simulation_steps = 28
num_cycles = 25
cycle_steps = 28
TRIALS = 1
INITIAL_IMMUNE_CELLS = 50
INITIAL_GOOD = 50
INITIAL_BAD = 20
INITIAL_GLYCOGEN = 2
IMMUNE_INTERACTION_RANGE = 0.1
GLYCOGEN_RANGE = 0.2

grid_size = 10
num_bins = 10
bin_size = 1  # Each bin is 10x10 units
num_bins = grid_size // bin_size


class Microbe:
    def __init__(self, species, location):
        self.species = species
        self.location = location

    def move(self, bacteria, alignment, index):
        new_location = [self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1)]

        if abs(new_location[0]) > 9:
            new_location[0] = 9
        elif new_location[0] < 0:
            new_location[0] = 0

        if abs(new_location[1]) > 9:
            if random.randint(1,10) != 1:
                bacteria[alignment]["Total"][alignment+f"_Bacteria_{index}"][-1] -= 1
                bacteria[alignment]["Cells"][alignment+f"_Bacteria_{index}"].remove(self)
        elif new_location[1] < 0:
            if random.randint(1,10) != 1:
                bacteria[alignment]["Total"][alignment+f"_Bacteria_{index}"][-1] -= 1
                bacteria[alignment]["Cells"][alignment+f"_Bacteria_{index}"].remove(self)

        self.location = new_location
        return self.location
        #print(self.location)

    def interact(self, total_bacteria_array, system, step):
        pass#hormone_level = (Oestrogen + progesterone) / 2


class Good_Bacteria(Microbe):
    def growth(self, bacteria, system, step):                        
        if system["pH"][step] < 4.5:
            a = next((x for x in system["Glycogen"]["Object"] if calculate_distance(x.location, self.location) <= GLYCOGEN_RANGE), None)
            if a != None:
                #print("h")
                if a.quantity > 0:#  and system["Glycogen"]["Total"][-1] > 0:
                    new_bacteria = Good_Bacteria(self.species, location=self.location)
                    #print(new_bacteria)
                    bacteria["Good"]["Cells"][self.species].append(new_bacteria)
                    ##print(len(bacteria["Good"]["Cells"][self.species]))
                    bacteria["Good"]["Total"][self.species][-1] += 1
                    ##print(bacteria["Good"]["Total"][self.species][-1])
                    #print(system["Glycogen"]["Total"][-1])
                    system["Glycogen"]["Total"][-1] -= 1
                    #print(system["Glycogen"]["Total"][-1])
                    index = system["Glycogen"]["Object"].index(a)
                    #print(system["Glycogen"]["Object"][index].quantity)
                    system["Glycogen"]["Object"][index].quantity -=1
                    #print(self.species)
                    #print(system["Glycogen"]["Object"][index].quantity)
                    #a.quantity -= 1
                    #print("hi")
        #print(bacteria["Good"]["Total"])
            

        

class Bad_Bacteria(Microbe):
    def growth(self, bacteria, system, step):
        inhibition_factor = system["pH"][step]
        #print(system["Iron"][step])                      
        if inhibition_factor > 4:
            for i in range(0,20):  
               a = next((x for x in system["Glycogen"]["Object"] if calculate_distance(x.location, self.location) <= GLYCOGEN_RANGE), None)
               if a != None:
                   #print("hi")
                   if a.quantity > 0:# and system["Glycogen"]["Total"][-1] > 0:
                       #print("hi")
                       #print(self.species)
                       new_bad_cell = Bad_Bacteria(self.species, self.location)
                       bacteria["Bad"]['Cells'][self.species].append(new_bad_cell)
                       bacteria["Bad"]["Total"][self.species][-1] += 1
                       #print(system["Glycogen"]["Total"][-1])
                       system["Glycogen"]["Total"][-1] -= 1
                       #print(system["Glycogen"]["Total"][-1])
                       index = system["Glycogen"]["Object"].index(a)
                       system["Glycogen"]["Object"][index].quantity -=1
                       

class Immune_Cells():
    def __init__(self, species, location, kills):
        self.species = species
        self.location = location
        self.kills = kills
        
    def move(self):
        new_location = [self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1)]  
        if abs(new_location[0]) > 9:
            new_location[0] = 9
        if new_location[0] < 0:
            new_location[0] = 0
            #should change
            #just temporary
            #they would survive outside top or bottom
        if abs(new_location[1]) > 9:
            new_location[1] = 9
        if abs(new_location[1]) < 0:
            new_location[1] = 0
            
        self.location = new_location
    
    def interact(self, immune_cells, bacteria, system, step, cycle, array):
        if immune_cells["Total"][-1] > 0: 
            for cell in array:
                #print(item.species)
                for i in range(1, 5):
                    #print(i)
                    #print(item.species)
                    if cell.species == f"Good_Bacteria_{i}":
                        if self in immune_cells["Cells"]:
                        #print("hi")
                            if cell in bacteria["Good"]["Cells"][cell.species]:
                                if calculate_distance(self.location, cell.location) <= IMMUNE_INTERACTION_RANGE:
                                    #print("yes")
                                    bacteria["Good"]["Cells"][cell.species].remove(cell)
                                    bacteria["Good"]["Total"][f"Good_Bacteria_{i}"][-1] -= 1
                                    self.kills += 1
                                    #print(self.kills)
                                    if self.kills > 1:
                                        #print(self)
                                        immune_cells['Cells'].remove(self)
                                        immune_cells["Total"][-1] -= 1
                                    
                    elif cell.species == f"Bad_Bacteria_{i}":
                        if self in immune_cells["Cells"]:
                            if cell in bacteria["Bad"]["Cells"][cell.species]:
                                #print("hi")
                                if calculate_distance(self.location, cell.location) <= IMMUNE_INTERACTION_RANGE:
                                    #print("yes")
                                    bacteria["Bad"]["Cells"][cell.species].remove(cell)
                                    bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"][-1] -= 1
                                    self.kills += 1
                                    #print(self.kills)
                                    if self.kills > 1:
                                        #print(self)
                                        immune_cells['Cells'].remove(self)
                                        immune_cells["Total"][-1] -= 1


class Environment_Location():
    def __init__(self, species, location, quantity):
        self.species = species
        self.location = location
        self.quantity = quantity
    
    def diffusion(self, system):
        for x,y in [(self.location[0]+i, self.location[1]+j) for i in (-1,0,1) for j in (-1,0,1) if i != 0 or j != 0]:
            a = next((c for c in system["Glycogen"]["Object"] if c.location == [x,y]), None)
            if a != None:
                if self.quantity > a.quantity:
                    index = system["Glycogen"]["Object"].index(a)
                    system["Glycogen"]["Object"][index].quantity += 1
                    self.quantity -= 1         
                    return index
        


def read_data():
    """
    Returns the data to be used
    -------
    Oestrogen_data : Array
        Array of Oestrogen values.
    progesterone_data : Array
        Array of progesterone values.
    maxE : Float
        Maximum Oestrogen value.
    maxP : Float
        Maximum progesterone value.
    day_no : Array
        Day hormones relate to.

    """
    data = np.genfromtxt("Tabhormone.csv", delimiter = ",", skip_header = 1)
    Oestrogen_data = data[:,2]
    day_no = data[:,0] + 17
    progesterone_data = data[:,5]
    maxE = np.max(Oestrogen_data)
    maxP = np.max(progesterone_data)
    return Oestrogen_data, progesterone_data, maxE, maxP, day_no


def iron_gen(system, step):
    hormone_level = (system["Oestrogen"][step] + system["Progesterone"][step])/2
    if  hormone_level < 0.2:
        iron = random.uniform(0.6, 0.9)
    else:
        iron = random.uniform(0.1, 0.4)
    #else:
     #   iron = random.uniform(0.05, 0.05)
    return iron


def glycogen_gen(system, step):
    hormone_level = (system["Oestrogen"][step] + system["Progesterone"][step]) / 2
    #glycogen levels depend on hormone levels
    new = 0
    if hormone_level > 0.4:
        additional = random.randint(40, 70)
            
    elif hormone_level > 0.2:
        additional = random.randint(10, 20)
        
    else:
        additional = random.randint(1, 6)
    for i in range(1, additional+1):
        #print(i, additional)
        loc = [random.randint(0,9), random.randint(0,9)]
        a = next((x for x in system["Glycogen"]["Object"] if x.location == loc), None)
        index = system["Glycogen"]["Object"].index(a)
        system["Glycogen"]["Object"][index].quantity += 1
        #new += 1
        
    amount = system["Glycogen"]["Total"][-1] + additional
    return amount


def calculate_distance(position1, position2):
    displacement = np.sqrt((position1[0] - position2[0])**2
                           +(position1[1] - position2[1])**2)
    #print(displacement)
    return displacement


def solution_pH_calc(pH1, pH2, vol1, vol2):
    M1 = 10**-pH1
    M2 = 10**-pH2
    mol1 = M1*vol1
    mol2 = M2*vol2
    mol = mol1 + mol2
    M = mol / (vol1+vol2)
    pH = -np.log10(M)
    return pH
    

def pH(system, bacteria, step):
    #during menstruation menstrual blood reduces
    if step < 7:
        #pH perturbing
        pH_p = random.uniform(7.3, 7.5)
        '''
        if len(system["pH"]) == 0:
            #pH secreted
            pH_s = random.uniform(3.5, 4.5)
        else:
            pH_s = system["pH"][-1]
        '''
        #pH vagina
        pH_v = random.uniform(3.5, 4.5)
        vol_p = 1*10**-3
        vol_v = 0.2*10**-3
        
    else:
        if system["Oestrogen"][step] > 0.5:
            pH_p = system["pH"][-1]
            pH_v = random.uniform(3.7, 4.5)
            vol_p = 1*10**-3
            vol_v = 0.2*10**-3
        else:
            pH_p = system["pH"][-1]
            pH_v = random.uniform(3.5, 4.5)
            vol_p = 1*10**-3
            vol_v = 0.2*10**-3
    
    new_pH = solution_pH_calc(pH_p, pH_v, vol_p, vol_v)        
    return new_pH 


def initial_state(system, Oestrogen, progesterone, step, bacteria):
    system["Oestrogen"].append(Oestrogen)
    system["Progesterone"].append(progesterone)
    system["Iron"].append(iron_gen(system, step))
    system["pH"].append(pH(system, bacteria, step))
    system["Glycogen"]["Total"].append(glycogen_gen(system, step))

            
def new_immune_cells(system, step, immune_cells):
    immune_cells["Total"].append(immune_cells["Total"][-1])
    if system["Oestrogen"][step] > 0.6:
        for i in range(1,3):
            new_immune_cell = Immune_Cells("Cells",(random.uniform(0, 9), random.uniform(0, 9)), 0)
            immune_cells['Cells'].append(new_immune_cell)
            immune_cells["Total"][-1] +=1
    
    elif system["Progesterone"][step] > 0.6:
        pass

    else:
        for i in range(0, 1):
            new_immune_cell = Immune_Cells("Cells",(random.uniform(0, 9), random.uniform(0, 9)), 0)
            immune_cells['Cells'].append(new_immune_cell)
            immune_cells["Total"][-1] += 1


def new_good_bacteria(system, bacteria, i):    
    if len(bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"]) == 0:
        bacteria["Good"]["Total"][f"Good_Bacteria_{i}"].append(INITIAL_GOOD)
        for x in range(0, INITIAL_GOOD):
            new_bacteria = Good_Bacteria(f"Good_Bacteria_{i}", location=(random.uniform(0, 9),
                                                                         random.uniform(0, 9)))
            bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"].append(new_bacteria)
            #bacteria["Good"]["Total"][f"Good_Bacteria_{i}"][-1] += 1
                
def create_good_bacteria(system, bacteria, step, item):
    #print("hi")
    Good_Bacteria.growth(item, bacteria, system, step)
    
        
def new_bad_bacteria(system, bacteria, i):
    if len(bacteria["Bad"]["Cells"][f"Bad_Bacteria_{i}"]) == 0:
        #print(i)
        bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"].append(INITIAL_BAD)
        for x in range(0, INITIAL_BAD):
            new_bacteria = Bad_Bacteria(f"Bad_Bacteria_{i}", location=(random.uniform(0, 9),
                                                                       random.uniform(0, 9)))
            bacteria["Bad"]["Cells"][f"Bad_Bacteria_{i}"].append(new_bacteria)
            #bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"][-1] += 1
            
def create_bad_bacteria(system, bacteria, step, item):
    Bad_Bacteria.growth(item, bacteria, system, step)
            #system["Iron"][step] = system["Iron"][step] - random.uniform(0.01, 0.035)
    
            #system["Iron"][step] = system["Iron"][step] - random.uniform(0.025, 0.052)


def good_action(system, step, immune_cells, total_bacteria, i):
    for cell in total_bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"]:
        cell.move(total_bacteria, "Good", i)
        #cell.interact(total_bacteria, system, step)
    
def bad_action(system, step, immune_cells, total_bacteria, i):
    for cell in total_bacteria["Bad"]["Cells"][f"Bad_Bacteria_{i}"]:
        cell.move(total_bacteria, "Bad", i)
        #cell.interact(total_bacteria, system, step)
        
    
def immune_action(system, step, immune_cells, total_bacteria, cycle, array):
    for cell in immune_cells["Cells"]:
        #print(step, cycle)
        cell.move()
        cell.interact(immune_cells, total_bacteria, system, step, cycle, array)


def get_bin(location):
    # Calculate the bin index, ensuring it's clamped within [0, num_bins - 1]
    x_bin = int(min(max(location[0] // bin_size, 0), num_bins - 1))
    y_bin = int(min(max(location[1] // bin_size, 0), num_bins - 1))
    return x_bin, y_bin

    
def create_heatmap(trackers, entity_type):
    # Initialize an empty grid with the correct number of bins
    heatmap = np.zeros((num_bins, num_bins))
    
    if entity_type == "Immune":
        # Loop over immune cells
        for immune_cell in trackers['Cells']:
            x_bin, y_bin = get_bin(immune_cell.location)
            heatmap[y_bin, x_bin] += 1
            
    elif entity_type == "Glycogen":
        # Loop over immune cells
        for glycogen_cell in trackers["Glycogen"]["Object"]:
            #print(glycogen_cell)
            x_bin, y_bin = get_bin(glycogen_cell.location)
            heatmap[y_bin, x_bin] = glycogen_cell.quantity

    else:
        # Loop over good bacteria
        for species in trackers[entity_type]["Cells"]:
            for bacterium in trackers[entity_type]["Cells"][species]:
                #print(bacterium)
                #print(species)
                x_bin, y_bin = get_bin(bacterium.location)
                # Increment count in the correct bin
                heatmap[y_bin, x_bin] += 1


    return heatmap


def plot_heatmap(heatmapgb, heatmapbb, heatmapi, heatmapg):
    plt.figure(figsize=(6, 6))
    plt.subplot(2,2,1)
    plt.imshow(heatmapgb, cmap="Greys", interpolation='nearest',
               origin='lower', extent=[0, grid_size, 0, grid_size])
    plt.colorbar(label='Count')
    plt.title("Good Bacteria")
    plt.xlabel('X Position ')
    plt.ylabel('Y Position')
    
    plt.subplot(2,2,2)
    plt.imshow(heatmapbb, cmap="Reds", interpolation='nearest',
               origin='lower', extent=[0, grid_size, 0, grid_size])
    plt.colorbar(label='Count')
    plt.title("Bad Bacteria")
    plt.xlabel('X Position ')
    plt.ylabel('Y Position')
    
    plt.subplot(2,2,3)
    plt.imshow(heatmapi, cmap="Blues", interpolation='nearest',
               origin='lower', extent=[0, grid_size, 0, grid_size])
    plt.colorbar(label='Count')
    plt.title("Immune_Cells")
    plt.xlabel('X Position ')
    plt.ylabel('Y Position')
    
    plt.subplot(2,2,4)
    plt.imshow(heatmapg, cmap="Greens", interpolation='nearest',
               origin='lower', extent=[0, grid_size, 0, grid_size])
    plt.colorbar(label='Count')
    plt.title("Glycogen")
    plt.xlabel('X Position ')
    plt.ylabel('Y Position')
    plt.tight_layout()
    
    plt.show()
    
    
def simulate_menstrual_cycle(num_cycles, cycle_steps):    
    Oestrogen, progesterone, maxE, maxP, day = read_data()    
    for trial in range(TRIALS):
        total_bacteria = {
            "Good": {"Cells": {},
             "Total": {}},
    
            "Bad": {"Cells": {},
            "Total": {}}
            }
        immune_cells = {
            "Cells": [Immune_Cells(species = "Cells", location=
                              (random.uniform(0, 9), random.uniform(0, 9)), kills = 0)
                      for i in range(INITIAL_IMMUNE_CELLS)],
            "Total": [INITIAL_IMMUNE_CELLS]
                }
        
        environment = {
            "Oestrogen": [],
            "Progesterone": [],
            "Iron": [],
            "Glycogen": {"Object": [Environment_Location("Glycogen", [i,j], INITIAL_GLYCOGEN)
                                    for i in range(0,10) for j in range(0,10)],
                         "Total": [INITIAL_GLYCOGEN*100]
                         },
            "pH": []
        }
        
        # Initialize abundance for each good bacteria species
        for i in range(1, num_microbes + 1):
            total_bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"] = []
            total_bacteria["Good"]["Total"][f"Good_Bacteria_{i}"] = []
        for i in range(1, 3):
            total_bacteria["Bad"]["Cells"][f"Bad_Bacteria_{i}"] = []
            total_bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"] = []
        
        for i in range(1, num_microbes +1):
            new_good_bacteria(environment, total_bacteria, i)
        for i in range(1, 3):
            new_bad_bacteria(environment, total_bacteria, i)
            
        for cycle in range(num_cycles):
            #running_good_total = []
            #running_bad_total = []
            for step in range(cycle_steps):
                Oestrogen_level = Oestrogen[step] / maxE
                progesterone_level = progesterone[step] / maxP
                initial_state(environment, Oestrogen_level, progesterone_level, step, total_bacteria)
                
                array = []
                for i in range(1, num_microbes +1):
                    array.extend(total_bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"])
                    total_bacteria["Good"]["Total"][f"Good_Bacteria_{i}"].append(total_bacteria["Good"]["Total"][f"Good_Bacteria_{i}"][-1])
                    if i < 3:
                        array.extend(total_bacteria["Bad"]["Cells"][f"Bad_Bacteria_{i}"])
                        total_bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"].append(total_bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"][-1])

                #print(array)
                random.shuffle(array)
                if step > 0:
                    for item in array:
                        #print(array.index(item))
                        #print(item.species)
                        for i in range(1, 5):
                            #print(item.species)
                            if item.species == f"Good_Bacteria_{i}":
                                create_good_bacteria(environment, total_bacteria, step, item)
                            elif item.species == f"Bad_Bacteria_{i}":
                                create_bad_bacteria(environment, total_bacteria, step, item)
                            
                for i in range(1, num_microbes+1):
                    good_action(environment, step, immune_cells, total_bacteria, i)
                
                for i in range(1,3):  
                    bad_action(environment, step, immune_cells, total_bacteria, i)
                    
                #for immune_cell in total_immune_cells:
                new_immune_cells(environment, step, immune_cells)
                immune_action(environment, step, immune_cells, total_bacteria, cycle, array)

                for cell in environment["Glycogen"]["Object"]:
                    Environment_Location.diffusion(cell, environment)

                
    days = np.max(day)*num_cycles
    plotting(environment, total_bacteria, immune_cells, days)
    good_bacteria_heat = create_heatmap(total_bacteria, "Good")
    bad_bacteria_heat = create_heatmap(total_bacteria, "Bad")
    immune_heat = create_heatmap(immune_cells, "Immune")
    glycogen_heat = create_heatmap(environment, "Glycogen")
    plot_heatmap(good_bacteria_heat, bad_bacteria_heat, immune_heat, glycogen_heat)



def plotting(system, bacteria, immune_cells, days):
    #print(system["Glycogen"])
    plt.figure(figsize=(12, 14))
    plot_no = 7
    
    plt.subplot(plot_no, 1, 1)
    plt.plot(range(1, len(system["Oestrogen"]) + 1), system["Oestrogen"], label='Oestrogen Levels',
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
    #print(bacteria)
    
    for good_species in bacteria["Good"]["Total"]:
        bacteria["Good"]["Total"][good_species].remove(bacteria["Good"]["Total"][good_species][0])
        plt.plot(range(1, len(bacteria["Good"]["Total"][good_species])+1),
                 bacteria["Good"]["Total"][good_species], label=f'{good_species}')
    for bad_species in bacteria["Bad"]["Total"]:
            #print(total_bad[bad_species])
        bacteria["Bad"]["Total"][bad_species].remove(bacteria["Bad"]["Total"][bad_species][0])
        plt.plot(range(1,len(bacteria["Bad"]["Total"][bad_species])+1),
                 bacteria["Bad"]["Total"][bad_species], label=f'{bad_species}')
   
    plt.title('Bacteria Abundance Over Time')
    plt.xlabel('Day')
    plt.ylabel('Abundance')
    plt.xlim(0,days)
    plt.legend(loc = "best")
    #print(len(bacteria["Good"]["Total"]["Good_Bacteria_4"]))
    #print(bacteria["Good"]["Total"]["Good_Bacteria_4"][-1])
    
    system["Glycogen"]["Total"].remove(system["Glycogen"]["Total"][0])
    #for x in system["Glycogen"]["Object"]:
    #    print(x.quantity)
    plt.subplot(plot_no, 1, 4)
    plt.plot(range(1, len(system["Glycogen"]["Total"]) + 1), system["Glycogen"]["Total"],
             label = "Glycogen Levels")
    plt.title('Glycogen Abundance Over Time')
    plt.xlabel('Day')
    plt.ylabel('Abundance')
    plt.xlim(0,days)
    plt.legend()
    
    plt.subplot(plot_no, 1, 5)
    plt.plot(range(1, len(system["pH"]) + 1), system["pH"],
             label='pH Levels', linestyle='--', color='green')
    plt.title('pH Levels')
    plt.xlabel('Day')
    plt.ylabel('pH')
    plt.xlim(0,days)
    plt.legend(loc = "best")

    plt.subplot(plot_no, 1, 6)
    plt.plot(range(1, len(immune_cells['Total']) + 1),
             immune_cells['Total'], label='Immune Cells',
             linestyle='--', color='green')
    plt.title('Total Immune Cells')
    plt.xlabel('Day')
    plt.ylabel('Abundance')
    plt.xlim(0,days)
    plt.legend(loc = "best")

    plt.tight_layout()
    plt.show()
    print("Execution completed.")


simulate_menstrual_cycle(num_cycles, cycle_steps)

