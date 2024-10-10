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

    # refresh all the levels for each trial as they are a new "Person"
    iron_levels = []
    estrogen_levels_over_time = []
    progesterone_levels_over_time = []
    hydrogen_peroxide_levels = []

    # Initialize bacteria trackers
    good_bacteria_tracker = {f"Good_Bacteria_{i}": [0] for i in range(1, 5)}
    bad_bacteria_tracker = {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)}

    # Track step-wise bacteria counts
    good_bacteria_step_tracker = {f"Good_Bacteria_{i}": [0] for i in range(1, 5)}
    bad_bacteria_step_tracker = {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)}
    good_bacteria_step_tracker_plot = {f"Good_Bacteria_{i}": [0] for i in range(1, 5)}
    bad_bacteria_step_tracker_plot = {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)}

    total_good_bacteria = []
    total_bad_bacteria = []
    total_bacteria = []

    # Initialize persistent microbes and immune cells
    microbes = [Microbe(species=f"Microbe_{i}", location=(random.uniform(0, 10), random.uniform(0, 10))) for i in range(NUM_MICROBES)]
    immune_cells = []

    # Immune cell trackers
    immune_kill_bad_tracker = []  # Immune cells that kill bad bacteria
    immune_kill_good_tracker = []  # Immune cells that kill good bacteria
    immune_kill_bad_count = 0
    immune_kill_good_count = 0
    good_bacteria_count = 0
    bad_bacteria_count = 0
    glycogen_levels = []

    remaining_immune_cells = []
    immune_count = 0
    immune_tracker = []

    for cycle in range(NUM_CYCLES):
        print(f"cycle {cycle}")
        for step in range(DAYS):

            # Get hormone levels
            estrogen_level = estrogen_levels[step]
            estrogen_levels_over_time.append(estrogen_level)
            progesterone_level = progesterone_levels[step]
            progesterone_levels_over_time.append(progesterone_level)

            # Calculate the total bacteria count
            total_bacteria_count = good_bacteria_count + bad_bacteria_count

            # Calculate iron and glycogen levels
            iron_level = iron_pulse(estrogen_level, progesterone_level)
            iron_levels.append(iron_level)

            glycogen_level = glycogen_production(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, max_bacteria_capacity)
            glycogen_levels.append(glycogen_level)

            # Reset step trackers
            for good_species in good_bacteria_step_tracker:
                good_bacteria_step_tracker[good_species] = 0
            for bad_species in bad_bacteria_step_tracker:
                bad_bacteria_step_tracker[bad_species] = 0

            # Microbes interact and move
            for microbe in microbes:
                microbe.move()
                microbe.interact(estrogen_level, progesterone_level, 1, iron_level, 0, 0, glycogen_level)

                if microbe.species is None:

                    continue

                elif "Good_Bacteria" in microbe.species:
                    # Update good bacteria step tracker
                    good_bacteria_step_tracker[microbe.species] += 1



                elif "Bad_Bacteria" in microbe.species:
                    # Update bad bacteria step tracker
                    bad_bacteria_step_tracker[microbe.species] += 1



            # Immune cells interact with bacteria and remove them after interaction

            for _ in range(number_immune_cells):
                if estrogen_level > 0.5:
                    target_type = random.choice(['kill_good','kill_bad'])
                    new_immune_cell = Immune_Cell(target_type, (random.uniform(0,10),random.uniform(0,10)))
                    remaining_immune_cells.append(new_immune_cell)

            for i in remaining_immune_cells:
                killed, good_bacteria_tracker, bad_bacteria_tracker = i.interact(good_bacteria_tracker, bad_bacteria_tracker)
                if killed == True:
                    remaining_immune_cells.remove(i)

            immune_tracker.append(len(remaining_immune_cells))


            # Append current step counts to trackers
            for good_species in good_bacteria_tracker:
                good_bacteria_tracker[good_species].append(good_bacteria_step_tracker[good_species] + good_bacteria_tracker[good_species][-1])
            for bad_species in bad_bacteria_tracker:
                bad_bacteria_tracker[bad_species].append(bad_bacteria_step_tracker[bad_species] + bad_bacteria_tracker[bad_species][-1])
            # Calculate total good and bad bacteria counts
            good_bacteria_count = sum(good_bacteria_tracker[species][-1] for species in good_bacteria_tracker)
            bad_bacteria_count = sum(bad_bacteria_tracker[species][-1] for species in bad_bacteria_tracker)
            total_bacteria_count = good_bacteria_count + bad_bacteria_count

            hydrogen_peroxide_level = h202_level(good_bacteria_count, bad_bacteria_count,estrogen_level)
            hydrogen_peroxide_levels.append(hydrogen_peroxide_level)
            total_good_bacteria.append(good_bacteria_count)
            total_bad_bacteria.append(bad_bacteria_count)
            total_bacteria.append(total_bacteria_count)

            for good_species in good_bacteria_step_tracker:
                good_bacteria_step_tracker_plot[good_species] = [good_bacteria_tracker[good_species][i+1] - good_bacteria_tracker[good_species][i] for i in range(len(good_bacteria_tracker[good_species])-1)]
            for bad_species in bad_bacteria_step_tracker:
                bad_bacteria_step_tracker_plot[bad_species] = [bad_bacteria_tracker[bad_species][i+1] - bad_bacteria_tracker[bad_species][i] for i in range(len(bad_bacteria_tracker[bad_species])-1)]


        # Calculate the proportion of good bacteria at the end of the trial
    good_bacteria_proportion = good_bacteria_count / (good_bacteria_count + bad_bacteria_count)
    good_bacteria_proportion_tracker.append(good_bacteria_proportion)

# Calculate and print the average proportion of good bacteria across all trials
print(f"Average Proportion of good bacteria over {TRIALS} TRIALS is: {np.average(good_bacteria_proportion_tracker):.2f}")

# Calculate and print the standard deviation of the proportion of good bacteria across
# Calculate and print the standard deviation of the proportion of good bacteria across all trials
standard_deviation_proportion = np.std(good_bacteria_proportion_tracker)
print(f"The standard deviation of the trials is: {standard_deviation_proportion:.2f}")

# Plotting the results
plt.figure(figsize=(12, 10))

plt.subplot(7, 1, 1)
plt.plot(range(1, len(estrogen_levels_over_time) + 1), estrogen_levels_over_time, label='Estrogen Levels', linestyle='--')
plt.plot(range(1, len(progesterone_levels_over_time) + 1), progesterone_levels_over_time, label='Progesterone Levels', linestyle='--')
plt.title('Hormone Levels Over Time')
plt.xlabel('Simulation Steps')
plt.ylabel('Hormone Levels')
plt.legend()

plt.subplot(7, 1, 2)
plt.plot(range(1, len(hydrogen_peroxide_levels) + 1), hydrogen_peroxide_levels, label='H2O2 Levels', linestyle='--', color='orange')
plt.plot(range(1, len(glycogen_levels) + 1), glycogen_levels, label='Gylcogen Level', linestyle='--', color='blue')
plt.title('Chemical Levels over time')
plt.xlabel('Simulation Steps')
plt.ylabel('Level')
plt.legend()

plt.subplot(7, 1, 3)
plt.plot(range(1, len(iron_levels) + 1), iron_levels, label='Iron Levels', linestyle='--', color='orange')
plt.title('Iron Pulse')
plt.xlabel('Simulation Steps')
plt.ylabel('Iron Levels')
plt.legend()

plt.subplot(7, 1, 4)
plt.plot(range(1, len(total_bad_bacteria) + 1), total_bad_bacteria, label='Bad Bacteria', linestyle='--')
plt.plot(range(1, len(total_good_bacteria) + 1), total_good_bacteria, label='Good Bacteria', linestyle='--')
plt.plot(range(1, len(total_bacteria) + 1), total_bacteria, label='Total Bacteria', linestyle='--')
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
for good_species in good_bacteria_step_tracker_plot :
    plt.plot(range(1, len(good_bacteria_step_tracker_plot[good_species]) + 1), good_bacteria_step_tracker_plot[good_species], label=f'{good_species} Count')
plt.title('Good Bacteria Produced at step')
plt.xlabel('Simulation Steps')
plt.ylabel('Count')
plt.legend()

plt.subplot(7, 1, 6)
for bad_species in bad_bacteria_step_tracker_plot:
    plt.plot(range(1, len(bad_bacteria_step_tracker_plot[bad_species]) + 1), bad_bacteria_step_tracker_plot[bad_species], label=f'{bad_species} Count')
plt.title('Bad Bacteria Produced at step')
plt.xlabel('Simulation Steps')
plt.ylabel('Count')
plt.legend()

plt.subplot(7, 1, 7)
plt.plot(range(1, len(immune_tracker) + 1), immune_tracker ,  label='Good Immune cells', linestyle='--')

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