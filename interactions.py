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
17/10/24
Started: 15:58
Finished:
"""

import random
import matplotlib.pyplot as plt
import numpy as np


NUM_MICROBES = 4
SIMULATION_STEPS = 28
NUM_CYCLES = 1
CYCLE_STEPS = 28
TRIALS = 1
INITIAL_IMMUNE_CELLS = 100
INITIAL_GOOD = 10
INITIAL_BAD = 10

class Microbe:
    def __init__(self, species, location):
        self.species = species
        self.location = location

    def move(self, bacteria, alignment, index):
        new_location = [self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1)]

        if abs(new_location[0]) > 10:
            new_location[0] = np.sign(new_location[0])*10

        if abs(new_location[1]) > 10:
            if random.randint(1,2) == 1:
                bacteria[alignment]["Total"][alignment+f"_Bacteria_{index}"][-1] -= 1
                bacteria[alignment]["Cells"][alignment+f"_Bacteria_{index}"].remove(self)

        self.location = new_location
        return self.location

    def interact(self, total_bacteria_array, system, step):
        pass#hormone_level = (estrogen + progesterone) / 2


class Good_Bacteria(Microbe):
    def growth(self, bacteria, system, step, i):
        if system["Hydrogen_Peroxide"][step] > 2:
            #print("hi")
            for x in range(0,2):
                a = next((x for x in system["Glycogen"]["Object"]
                          if x.location == [round(self.location[0]),
                                            round(self.location[1])]), None)
                if a != None:
                    if a.quantity > 0:
                        new_bacteria = Good_Bacteria(f"Good_Bacteria_{i}"
                                                     , location=self.location)
                        bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"].append(new_bacteria)
                        bacteria["Good"]["Total"][f"Good_Bacteria_{i}"][-1]+=1
                        system["Glycogen"]["Total"][-1] -= 1
                        index = system["Glycogen"]["Object"].index(a)
                        system["Glycogen"]["Object"][index].quantity -=1
    #def interact(self, total_bacteria_array, system, step):


class Bad_Bacteria(Microbe):
    def growth(self, bacteria, system, step, i):
        inhibition_factor = 0.2
        if system["Iron"][step] > inhibition_factor:
            for x in range(0,1):
                a = next((x for x in system["Glycogen"]["Object"]
                          if x.location == [round(self.location[0])
                                            ,round(self.location[1])]), None)
                if a != None:
                    if a.quantity > 0:
                        new_bad_cell = Bad_Bacteria(f"Bad_Bacteria_{i}", self.location)
                        bacteria["Bad"]['Cells'][f"Bad_Bacteria_{i}"].append(new_bad_cell)
                        bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"][-1] += 1
                        system["Glycogen"]["Total"][-1] -= 1
                        index = system["Glycogen"]["Object"].index(a)
                        system["Glycogen"]["Object"][index].quantity -=1

        elif system["Iron"][step] > 0.5:
            for x in range(1, 2):
                a = next((x for x in system["Glycogen"]["Object"]
                          if x.location == [round(self.location[0]),
                                            round(self.location[1])]), None)
                if a != None:
                    if a.quantity > 0:
                        new_bad_cell = Bad_Bacteria("Cells", self.location)
                        bacteria["Bad"]['Cells'][f"Bad_Bacteria_{i}"].append(new_bad_cell)
                        bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"] += 1
                        system["Glycogen"]["Total"][-1] -= 1
                        index = system["Glycogen"]["Object"].index(a)
                        system["Glycogen"]["Object"][index].quantity -=1

    #def interact(self, total_bacteria_array, system, step):


class Immune_Cells():
    def __init__(self, species, location, kills):
        self.species = species
        self.location = location
        self.kills = kills

    def move(self):
        new_location = [self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1)]
        if abs(new_location[0]) > 10:
            new_location[0] = np.sign(new_location[0])*10
        self.location = new_location

    def interact(self, immune_cells, bacteria, system, step):
        if immune_cells["Total"][-1] > 0:
            for i in range(1, 3):
                b = bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"]
                for cell in b:
                    if self in immune_cells["Cells"]:
                        if self.location[0] <= cell.location[0] + 1 and self.location[0] >= cell.location[0] - 1:
                            if self.location[1] <= cell.location[1] + 1 and self.location[1] >= cell.location[1] - 1:
                                b.remove(cell)
                                bacteria["Good"]["Total"][f"Good_Bacteria_{i}"][-1] -= 1
                                self.kills += 1
                                #print(self.kills)
                                if self.kills > 1:
                                    #print(self)
                                    immune_cells['Cells'].remove(self)
                                    immune_cells["Total"][-1] -= 1

            for i in range(1,2):
                if self in immune_cells["Cells"]:
                    b = bacteria["Bad"]["Cells"][f"Bad_Bacteria_{i}"]
                    for cell in b:
                        if self in immune_cells["Cells"]:
                            if self.location[0] <= cell.location[0] + 1 and self.location[0] >= cell.location[0] - 1:
                                if self.location[1] <= cell.location[1] + 1 and self.location[1] >= cell.location[1] - 1:
                                    b.remove(cell)
                                    bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"][-1] -= 1
                                    self.kills += 1
                                    #print(self.kills)
                                    if self.kills > 1:
                                        immune_cells['Cells'].remove(self)
                                        immune_cells["Total"][-1] -= 1


class Environment_Location():
    def __init__(self, species, location, quantity):
        self.species = species
        self.location = location
        self.quantity = quantity


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
    #new_glycogen = Glycogen()

    if len(system["Glycogen"]["Total"]) == 0:
        amount = random.randint(2,4)
        for i in range(0, amount):
            loc = [random.randint(0,10), random.randint(0,10)]
            a = next((x for x in system["Glycogen"]["Object"] if x.location == loc), None)
            if a != None:
                a.quantity +=1

    else:
        #print(system["Glycogen"])
        if hormone_level > 0.4:
            additional = random.randint(5, 9)
            for i in range(1, additional):
                loc = [random.randint(0,10), random.randint(0,10)]
                a = next((x for x in system["Glycogen"]["Object"] if x.location == loc), None)
                if a != None:
                    a.quantity +=1
        elif hormone_level > 0.2:
            additional = random.randint(5, 9)
            for i in range(1, additional):
                loc = [random.randint(0,10), random.randint(0,10)]
                a = next((x for x in system["Glycogen"]["Object"] if x.location == loc), None)
                if a != None:
                    a.quantity +=1
        else:
            additional = random.randint(5, 9)
            for i in range(1, additional):
                loc = [random.randint(0,10), random.randint(0,10)]
                a = next((x for x in system["Glycogen"]["Object"] if x.location == loc), None)
                if a != None:
                    a.quantity +=1
        amount = system["Glycogen"]["Total"][-1] + additional
    return amount


def hydrogen_peroxide_gen(system, total_good_bacteria, total_bad_bacteria, step):
    if len(system["Hydrogen_Peroxide"]) == 0:
        hydrogen_peroxide = total_good_bacteria
    elif step < 7:
        additional_hydrogen_peroxide = - random.uniform(0.5,1)
    else:
        if total_good_bacteria*0.4 > total_bad_bacteria:
            additional_hydrogen_peroxide = random.uniform(0.3,0.6)
        else:
            additional_hydrogen_peroxide = random.uniform(0.1,0.4)
    hydrogen_peroxide = max((system["Hydrogen_Peroxide"][-1] + additional_hydrogen_peroxide),0)

    return hydrogen_peroxide


def initial_state(system, estrogen, progesterone, step, good_bacteria_array, bad_bacteria_array):
    system["Estrogen"].append(estrogen)
    system["Progesterone"].append(progesterone)
    system["Iron"].append(iron_gen(system, step))
    system["Hydrogen_Peroxide"].append(hydrogen_peroxide_gen(system,
                                                    sum(good_bacteria_array),
                                                    sum(bad_bacteria_array),step))
    system["Glycogen"]["Total"].append(max(0,glycogen_gen(system, step)))


def new_immune_cells(system, step, immune_cells):
    immune_cells["Total"].append(immune_cells["Total"][-1])
    if system["Estrogen"][step] > 0.6:
        for i in range(2, round(immune_cells["Total"][-1]*0.1)):
            new_immune_cell = Immune_Cells("Cells",(random.uniform(0, 10), random.uniform(0, 10)), 0)
            immune_cells['Cells'].append(new_immune_cell)
            immune_cells["Total"][-1] +=1

    else:
        for i in range(2, round(immune_cells["Total"][-1]*0.07)):
            new_immune_cell = Immune_Cells("Cells",(random.uniform(0, 10), random.uniform(0, 10)), 0)
            immune_cells['Cells'].append(new_immune_cell)
            immune_cells["Total"][-1] += 1


def new_good_bacteria(system, bacteria, step, i):
    #print(bacteria)
    bacteria["Good"]["Total"][f"Good_Bacteria_{i}"].append(bacteria["Good"]["Total"][f"Good_Bacteria_{i}"][-1])
    if len(bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"]) == 0:
        for x in range(1, INITIAL_GOOD):
            new_bacteria = Good_Bacteria(f"Good_Bacteria_{i}", location=(random.uniform(0, 10),
                                                                         random.uniform(0, 10)))
            bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"].append(new_bacteria)
            bacteria["Good"]["Total"][f"Good_Bacteria_{i}"][-1] += 1

    else:
        for x in bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"]:
            Good_Bacteria.growth(x, bacteria, system, step, i)


def new_bad_bacteria(system, bacteria, step, i):
    bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"].append(bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"][-1])
    if len(bacteria["Bad"]["Cells"][f"Bad_Bacteria_{i}"]) ==0:
        for x in range(1, INITIAL_BAD):
            new_bacteria = Bad_Bacteria(f"Bad_Bacteria_{i}", location=(random.uniform(0, 10),
                                                                       random.uniform(0, 10)))
            bacteria["Bad"]["Cells"][f"Bad_Bacteria_{i}"].append(new_bacteria)
            bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"][-1] += 1
    else:
        for x in bacteria["Bad"]["Cells"][f"Bad_Bacteria_{i}"]:
            Bad_Bacteria.growth(x, bacteria, system, step, i)
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


def immune_action(system, step, immune_cells, total_bacteria):
    for cell in immune_cells["Cells"]:
        cell.move()
        cell.interact(immune_cells, total_bacteria, system, step)


def simulate_menstrual_cycle(NUM_CYCLES, CYCLE_STEPS):
    estrogen, progesterone, maxE, maxP, day = read_data()
    for trial in range(TRIALS):
        total_bacteria = {
            "Good": {"Cells": {},
             "Total": {}},

            "Bad": {"Cells": {},
            "Total": {}}
            }
        immune_cells = {
            "Cells": [Immune_Cells(species = "Cells", location=
                              (random.uniform(0, 10), random.uniform(0, 10)), kills = 0)
                      for i in range(INITIAL_IMMUNE_CELLS)],
            "Total": [INITIAL_IMMUNE_CELLS]
                }
        good_bacteria_array = []
        bad_bacteria_array = []

        environment = {
            "Estrogen": [],
            "Progesterone": [],
            "Iron": [],
            "Glycogen": {"Object": [Environment_Location("Glycogen", [i,j], 0)
                                    for i in range(1,10) for j in range(1,10)],
                         "Total": []
                         },
            "Hydrogen_Peroxide": []
        }
        #print(environment["Glycogen"]["Object"][10].location)

        #microbes = [Microbe(species=f"Microbe_{i}", location=(random.uniform(
         #   0, 10), random.uniform(0, 10))) for i in range(NUM_MICROBES)]
        # Initialize abundance for each good bacteria species
        for i in range(1, NUM_MICROBES + 1):
            total_bacteria["Good"]["Cells"][f"Good_Bacteria_{i}"] = []
            total_bacteria["Good"]["Total"][f"Good_Bacteria_{i}"] = [0]
        for i in range(1, 3):
            total_bacteria["Bad"]["Cells"][f"Bad_Bacteria_{i}"] = []
            total_bacteria["Bad"]["Total"][f"Bad_Bacteria_{i}"] = [0]

        for cycle in range(NUM_CYCLES):
            #running_good_total = []
            #running_bad_total = []
            for step in range(CYCLE_STEPS):
                estrogen_level = estrogen[step] / maxE
                progesterone_level = progesterone[step] / maxP
                initial_state(environment, estrogen_level, progesterone_level, step, good_bacteria_array, bad_bacteria_array)


                for i in range(1, NUM_MICROBES+1):
                    new_good_bacteria(environment, total_bacteria, step, i)
                    good_action(environment, step, immune_cells, total_bacteria, i)

                for i in range(1,3):
                    new_bad_bacteria(environment, total_bacteria, step, i)
                    bad_action(environment, step, immune_cells, total_bacteria, i)

                #for immune_cell in total_immune_cells:
                new_immune_cells(environment, step, immune_cells)
                immune_action(environment, step, immune_cells, total_bacteria)

    total_days = np.max(day)*NUM_CYCLES
    plotting(environment, total_bacteria, immune_cells, total_days)


def plotting(system, bacteria, immune_cells, days):
    plt.figure(figsize=(12, 14))
    plot_no = 7

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
    for good_species in bacteria["Good"]["Total"]:
        plt.plot(range(1, len(bacteria["Good"]["Total"][good_species]) + 1),
                 bacteria["Good"]["Total"][good_species], label=f'{good_species} Abundance')
    for bad_species in bacteria["Bad"]["Total"]:
            #print(total_bad[bad_species])
        plt.plot(range(1,len(bacteria["Bad"]["Total"][bad_species])+1),
                 bacteria["Bad"]["Total"][bad_species], label=f'{bad_species} Abundance')

    plt.title('Bacteria Abundance Over Time')
    plt.xlabel('Day')
    plt.ylabel('Abundance')
    plt.xlim(0,days)
    plt.legend()

    plt.subplot(plot_no, 1, 4)
    plt.plot(range(1, len(system["Glycogen"]["Total"]) + 1), system["Glycogen"]["Total"],
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
