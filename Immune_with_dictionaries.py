#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:08:34 2024

@author: matthewillsley
"""

import random
import matplotlib.pyplot as plt
import numpy as np

NUM_MICROBES = 100
DAYS = 28
NUM_CYCLES = 10
TRIALS = 1 # You can increase the number of trials
prob_good_bacteria_create_immune = 0.05  # 5% chance for good bacteria to create immune cells
prob_bad_bacteria_create_immune = 0.05 #90% chance for bad bacteria to create immune cells

prob_kills_bad = 0.1 # 30% probability that good immune cells kill a bad bacteria
prob_kills_good = 0.2  # 90% probability that bad immune cells kill a good bacteria

max_bacteria_capacity = 20000

number_immune_cells = 60

# Microbe Class
class Microbe:
    def __init__(self, species, location):
        self.species = species
        self.location = location

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1), self.location[1] + random.uniform(-1, 1))
        self.location = new_location

    def interact(self, estrogen_level, progesterone_level, hydrogen_peroxide_level, iron_level, good_bacteria_count, bad_bacteria_count, glycogen_level):
        hormone_level = (estrogen_level + progesterone_level) / 2

        # Strong dependency on glycogen for good bacteria
        if (hormone_level > 0.5 or hydrogen_peroxide_level >= 2 * iron_level) and glycogen_level > 0:
            self.species = random.choice(["Good_Bacteria_1", "Good_Bacteria_2", "Good_Bacteria_3", "Good_Bacteria_4"])
        elif glycogen_level>0:
            self.species = random.choice(["Bad_Bacteria_1", "Bad_Bacteria_2"])
        else:
            self.species = None

# Immune Cell Class
class Immune_Cell:
    def __init__(self, target_type, location):
        self.target_type = target_type  # Can be 'kill_good' or 'kill_bad'
        self.location = location

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1), self.location[1] + random.uniform(-1, 1))
        self.location = new_location

    def interact(self, good_bacteria_tracker, bad_bacteria_tracker):
        if self.target_type == 'kill_bad':
            key = random.choice(list(bad_bacteria_tracker.keys()))
            if bad_bacteria_tracker[key][-1] > 0 and random.uniform(0,1)<prob_kills_bad:
                #print(key, "before", bad_bacteria_tracker[key][-1])
                bad_bacteria_tracker[key][-1] -= 1 # Kill one bad bacterium
                #print(key, "after", bad_bacteria_tracker[key][-1])
                return True, good_bacteria_tracker, bad_bacteria_tracker   # Immune cell dies after killing
        elif self.target_type == 'kill_good':
            key = random.choice(list(good_bacteria_tracker.keys()))
            if good_bacteria_tracker[key][-1] > 0 and random.uniform(0,1)<prob_kills_good:
                #print(key, "before", good_bacteria_tracker[key][-1])
                good_bacteria_tracker[key][-1] -= 1
                #print(key, "after",good_bacteria_tracker[key][-1])# Kill one good bacterium
                return True , good_bacteria_tracker, bad_bacteria_tracker # Immune cell dies after killing
        return False, good_bacteria_tracker, bad_bacteria_tracker  # Immune cell didn't kill anything, stays alive


# Read data function
def read_data():
    hormone_data = np.genfromtxt("TabHormone.csv", comments='%', delimiter=",", skip_header=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    day_no = hormone_data[:, 0]
    est_level = hormone_data[:, 2]
    progest_level = hormone_data[:, 5]
    return day_no, est_level, progest_level


# Iron pulse function
def iron_pulse(estrogen_level, progesterone_level):
    if estrogen_level < 0.2 and progesterone_level < 0.2:
        iron_level = random.uniform(0.7, 1)
    else:
        iron_level = random.uniform(0.1, 0.3)
    return iron_level


# H202 level function
def h202_level(good_bacteria_count, bad_bacteria_count,estrogen_level):
    if good_bacteria_count > 2*bad_bacteria_count and estrogen_level >0.7:
        return random.uniform(0.7, 1)
    elif good_bacteria_count > bad_bacteria_count and estrogen_level>0.2:
        return random.uniform(0.4,0.7)
    elif estrogen_level < 0.2:
        return random.uniform(0.1, 0.4)
    else:
        return random.uniform(0.4,.7)

def glycogen_production(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, max_bacteria_capacity):
    hormone_level = (estrogen_level + progesterone_level) / 2
    total_bacteria_count = good_bacteria_count + bad_bacteria_count
    capacity_usage = total_bacteria_count / max_bacteria_capacity

    if capacity_usage >= 1:
        glycogen_level = 0  # No glycogen left if we are at or over capacity
    elif capacity_usage > 0.7:
        # Rapid drop in glycogen as we approach capacity
        glycogen_level = random.uniform(0, 0.3) * (1 - capacity_usage) * hormone_level
    else:
        # Normal glycogen production but reduced as bacteria count grows
        glycogen_level = random.uniform(0.5, 1) * (1 - capacity_usage) * hormone_level

    return glycogen_level


def plotting(enviroment,microbe_trackers):

    # Plotting the results
    plt.figure(figsize=(12, 10))

    plt.subplot(7, 1, 1)
    plt.plot(range(1, len(enviroment["estrogen"]) + 1), enviroment["estrogen"], label='Estrogen Levels', linestyle='--')
    plt.plot(range(1, len(enviroment["progesterone"]) + 1), enviroment["progesterone"], label='Progesterone Levels', linestyle='--')
    plt.title('Hormone Levels Over Time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Hormone Levels')
    plt.legend()

    plt.subplot(7, 1, 2)
    plt.plot(range(1, len(enviroment["h2o2"]) + 1), enviroment["h2o2"], label='H2O2 Levels', linestyle='--', color='orange')
    plt.plot(range(1, len(enviroment["glycogen"]) + 1), enviroment["glycogen"], label='Gylcogen Level', linestyle='--', color='blue')
    plt.title('Chemical Levels over time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Level')
    plt.legend()

    plt.subplot(7, 1, 3)
    plt.plot(range(1, len(enviroment["iron"]) + 1), enviroment["iron"], label='Iron Levels', linestyle='--', color='orange')
    plt.title('Iron Pulse')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Iron Levels')
    plt.legend()

    plt.subplot(7, 1, 4)
    plt.plot(range(1, len(microbe_trackers["total bad"]) + 1), microbe_trackers["total bad"], label='Bad Bacteria', linestyle='--')
    plt.plot(range(1, len(microbe_trackers["total good"]) + 1), microbe_trackers["total good"], label='Good Bacteria', linestyle='--')
    plt.plot(range(1, len(microbe_trackers["total bacteria"]) + 1), microbe_trackers["total bacteria"] , label='Total Bacteria', linestyle='--')
    plt.title('Total Microbes')
    plt.xlabel('Simulation Steps')
    plt.ylabel('# of Bacteria')
    plt.legend()
    '''
    plt.subplot(7, 1, 5)
    for good_species in good_bacteria_tracker:
        plt.plot(range(1, len(good_bacteria_tracker[good_species]) + 1), good_bacteria_tracker[good_species], label=f'{good_species} Count')
    plt.title('Good Bacteria Produced Totals')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(7, 1, 6)
    for bad_species in bad_bacteria_tracker:
        plt.plot(range(1, len(bad_bacteria_tracker[bad_species]) + 1), bad_bacteria_tracker[bad_species], label=f'{bad_species} Count')
    plt.title('Bad Bacteria Produced Totals')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Count')
    plt.legend()'''

    plt.subplot(7, 1, 5)
    for good_species in microbe_trackers["good bacteria step plot"] :
        plt.plot(range(1, len(microbe_trackers["good bacteria step plot"][good_species]) + 1), microbe_trackers["good bacteria step plot"][good_species], label=f'{good_species} Count')
    plt.title('Good Bacteria Produced at step')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(7, 1, 6)
    for bad_species in microbe_trackers["bad bacteria step plot"]:
        plt.plot(range(1, len(microbe_trackers["bad bacteria step plot"][bad_species]) + 1), microbe_trackers["bad bacteria step plot"][bad_species], label=f'{bad_species} Count')
    plt.title('Bad Bacteria Produced at step')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(7, 1, 7)
    for immune_type in microbe_trackers["immune plot"] :
        plt.plot(range(1, len(microbe_trackers["immune plot"][immune_type]) + 1), microbe_trackers["immune plot"][immune_type], label=f'{immune_type} Count')
    plt.title('Immune cells')
    plt.xlabel('Simulation Steps')
    plt.ylabel('# of immune cells')
    plt.legend()

    '''plt.subplot(7, 1, 7)
    plt.hist(good_bacteria_proportion_tracker, bins=20, density=False, label="Distribution of Trials")
    plt.title('Histogram of Good Bacteria for Each Trial')
    plt.xlabel('Proportion of Good Bacteria')
    plt.ylabel('Counts')
    plt.legend()'''

    plt.tight_layout()
    plt.savefig(fname="plots", dpi=100)
    plt.show()

def simulation_loop():
    # Main simulation logic
    good_bacteria_proportion_tracker = []

    for test in range(TRIALS):
        print("Trial:", test + 1)

        # Read data
        data = read_data()
        estrogen_levels_raw = data[1]
        estrogen_levels = estrogen_levels_raw / max(estrogen_levels_raw)
        progesterone_levels_raw = data[2]
        progesterone_levels = progesterone_levels_raw / max(progesterone_levels_raw)

        # Initialize the enviroment dictionary to store hormone levels
        enviroment = {
            "estrogen": [],
            "progesterone": [],
            "iron": [],
            "glycogen": [],
            "h2o2": []
        }

        # Initialize bacteria trackers
        microbe_trackers = {
            "good bacteria tracker": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
            "bad bacteria tracker": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)},
            "good bacteria step": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
            "bad bacteria step": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)},
            "good bacteria step plot": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
            "bad bacteria step plot": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)},
            "immune cells": {'kill_good': [], 'kill_bad': []}
        }

        total_good_bacteria = []
        total_bad_bacteria = []
        total_bacteria = []

        # Initialize persistent microbes
        microbes = [Microbe(species=f"Microbe_{i}", location=(random.uniform(0, 10), random.uniform(0, 10))) for i in range(NUM_MICROBES)]

        good_bacteria_count = 0
        bad_bacteria_count = 0
        glycogen_levels = []

        immune_tracker = {'kill_good': [], 'kill_bad': []}

        for cycle in range(NUM_CYCLES):
            print(f"cycle {cycle}")
            for step in range(DAYS):

                # Get hormone levels
                estrogen_level = estrogen_levels[step]
                enviroment["estrogen"].append(estrogen_level)
                progesterone_level = progesterone_levels[step]
                enviroment["progesterone"].append(progesterone_level)

                # Calculate iron and glycogen levels
                iron_level = iron_pulse(estrogen_level, progesterone_level)
                enviroment["iron"].append(iron_level)

                glycogen_level = glycogen_production(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, max_bacteria_capacity)
                enviroment["glycogen"].append(glycogen_level)

                # Reset step trackers
                for good_species in microbe_trackers["good bacteria step"]:
                    microbe_trackers["good bacteria step"][good_species] = 0
                for bad_species in microbe_trackers["bad bacteria step"]:
                    microbe_trackers["bad bacteria step"][bad_species] = 0

                # Microbes interact and move
                for microbe in microbes:
                    microbe.move()
                    microbe.interact(estrogen_level, progesterone_level, 1, iron_level, 0, 0, glycogen_level)

                    if microbe.species is None:
                        continue
                    elif "Good_Bacteria" in microbe.species:
                        microbe_trackers["good bacteria step"][microbe.species] += 1
                    elif "Bad_Bacteria" in microbe.species:
                        microbe_trackers["bad bacteria step"][microbe.species] += 1

                # New Immune cells made based on conditions
                for _ in range(number_immune_cells):
                    if estrogen_level > 0.5:
                        target_type = random.choice(['kill_good', 'kill_bad'])
                        new_immune_cell = Immune_Cell(target_type, (random.uniform(0, 10), random.uniform(0, 10)))
                        microbe_trackers['immune cells'][target_type].append(new_immune_cell)

                # Immune cells interact
                for immune_type in microbe_trackers['immune cells']:
                    for i in microbe_trackers['immune cells'][immune_type]:
                        killed, good_bacteria_tracker, bad_bacteria_tracker = i.interact(microbe_trackers["good bacteria tracker"], microbe_trackers["bad bacteria tracker"])
                        if killed:
                            microbe_trackers['immune cells'][immune_type].remove(i)

                    immune_tracker[immune_type].append(len(microbe_trackers['immune cells'][immune_type]))

                # Append current step counts to trackers
                for good_species in microbe_trackers["good bacteria tracker"]:
                    microbe_trackers["good bacteria tracker"][good_species].append(microbe_trackers["good bacteria step"][good_species] + microbe_trackers["good bacteria tracker"][good_species][-1])
                for bad_species in microbe_trackers["bad bacteria tracker"]:
                    microbe_trackers["bad bacteria tracker"][bad_species].append(microbe_trackers["bad bacteria step"][bad_species] + microbe_trackers["bad bacteria tracker"][bad_species][-1])

                # Calculate total good and bad bacteria counts
                good_bacteria_count = sum(microbe_trackers["good bacteria tracker"][species][-1] for species in microbe_trackers["good bacteria tracker"])
                bad_bacteria_count = sum(microbe_trackers["bad bacteria tracker"][species][-1] for species in microbe_trackers["bad bacteria tracker"])
                total_bacteria_count = good_bacteria_count + bad_bacteria_count

                for good_species in microbe_trackers["good bacteria step plot"]:
                   microbe_trackers["good bacteria step plot"][good_species] =\
                       [microbe_trackers["good bacteria tracker"][good_species][i+1] - microbe_trackers["good bacteria tracker"][good_species][i] for i in range(len(microbe_trackers["good bacteria tracker"][good_species])-1)]
                for bad_species in microbe_trackers["bad bacteria step plot"]:
                   microbe_trackers["bad bacteria step plot"][bad_species] =\
                       [microbe_trackers["bad bacteria tracker"][bad_species][i+1] - microbe_trackers["bad bacteria tracker"][bad_species][i] for i in range(len(microbe_trackers["bad bacteria tracker"][bad_species])-1)]


                hydrogen_peroxide_level = h202_level(good_bacteria_count, bad_bacteria_count, estrogen_level)
                enviroment["h2o2"].append(hydrogen_peroxide_level)
                total_good_bacteria.append(good_bacteria_count)
                total_bad_bacteria.append(bad_bacteria_count)
                total_bacteria.append(total_bacteria_count)

        # Store the final environment and microbe trackers
        good_bacteria_proportion = good_bacteria_count / (good_bacteria_count + bad_bacteria_count)
        good_bacteria_proportion_tracker.append(good_bacteria_proportion)

        # Add total good and total bad bacteria to the microbe_trackers
        microbe_trackers["total good"] = total_good_bacteria
        microbe_trackers["total bad"] = total_bad_bacteria
        microbe_trackers["total bacteria"] = total_bacteria
        microbe_trackers["immune plot"] = immune_tracker

    return enviroment, microbe_trackers, good_bacteria_proportion_tracker




enviroment, microbe_trackers, good_bacteria_proportion_tracker = simulation_loop()
plotting(enviroment,microbe_trackers)


# Calculate and print the average proportion of good bacteria across all trials
print(f"Average Proportion of good bacteria over {TRIALS} TRIALS is: {np.average(good_bacteria_proportion_tracker):.2f}")

# Calculate and print the standard deviation of the proportion of good bacteria across
# Calculate and print the standard deviation of the proportion of good bacteria across all trials
standard_deviation_proportion = np.std(good_bacteria_proportion_tracker)
print(f"The standard deviation of the trials is: {standard_deviation_proportion:.2f}")

