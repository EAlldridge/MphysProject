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
TRIALS = 1  # You can increase the number of trials

PROB_KILL_GOOD_MADE = 0.4
PROB_KILL_BAD_MADE = 0.6

PROB_KILLS_BAD = 1  # 30% probability that good immune cells kill a bad bacteria
PROB_KILLS_GOOD = 0.1  # 90% probability that bad immune cells kill a good bacteria

MAX_BACTERIA_CAPACITY = 20000

NUMBER_IMMUNE_CELLS = 100

IMMUNE_CELL_AGE_LIMIT = DAYS * 2

# Microbe Class


class Microbe:
    def __init__(self, species, location):
        self.species = species
        self.location = location

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1))
        self.location = new_location

    def interact(self, estrogen_level, progesterone_level, hydrogen_peroxide_level, iron_level, glycogen_level):
        hormone_level = (estrogen_level + progesterone_level) / 2

        # Strong dependency on glycogen for good bacteria
        if hydrogen_peroxide_level >= 2 * iron_level and glycogen_level > 0:
            self.species = random.choice(
                ["Good_Bacteria_1", "Good_Bacteria_2", "Good_Bacteria_3", "Good_Bacteria_4"])
        elif glycogen_level > 0:
            self.species = random.choice(["Bad_Bacteria_1", "Bad_Bacteria_2"])
        else:
            self.species = None

# Immune Cell Class


class Immune_Cell:
    def __init__(self, target_type, location, age):
        self.target_type = target_type  # Can be 'kill_good' or 'kill_bad'
        self.location = location
        self.age = age

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1))
        self.location = new_location

    def interact(self, good_bacteria_tracker, bad_bacteria_tracker):
        self.age += 1
        if self.age > IMMUNE_CELL_AGE_LIMIT:
            print("immune too old")
            return True, good_bacteria_tracker, bad_bacteria_tracker
        if self.target_type == 'kill_bad':
            key = random.choice(list(bad_bacteria_tracker.keys()))
            if bad_bacteria_tracker[key][-1] > 0 and random.uniform(0, 1) < PROB_KILLS_BAD:
                #print(key, "before", bad_bacteria_tracker[key][-1])
                bad_bacteria_tracker[key][-1] -= 1  # Kill one bad bacterium
                #print(key, "after", bad_bacteria_tracker[key][-1])
                # Immune cell dies after killing
                return True, good_bacteria_tracker, bad_bacteria_tracker
        elif self.target_type == 'kill_good':
            key = random.choice(list(good_bacteria_tracker.keys()))
            if good_bacteria_tracker[key][-1] > 0 and random.uniform(0, 1) < PROB_KILLS_GOOD:
                #print(key, "before", good_bacteria_tracker[key][-1])
                good_bacteria_tracker[key][-1] -= 1
                # print(key, "after",good_bacteria_tracker[key][-1])# Kill one good bacterium
                # Immune cell dies after killing
                return True, good_bacteria_tracker, bad_bacteria_tracker
        # Immune cell didn't kill anything, stays alive
        return False, good_bacteria_tracker, bad_bacteria_tracker


# Read data function
def read_data():
    hormone_data = np.genfromtxt("TabHormone.csv", comments='%',
                                 delimiter=",", skip_header=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
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
def h202_level(good_bacteria_count, bad_bacteria_count, estrogen_level):
    if good_bacteria_count > 2*bad_bacteria_count and estrogen_level > 0.7:
        return random.uniform(0.7, 1)
    elif good_bacteria_count > bad_bacteria_count and estrogen_level > 0.2:
        return random.uniform(0.4, 0.7)
    elif estrogen_level < 0.2:
        return random.uniform(0.1, 0.4)
    else:
        return random.uniform(0.4, .7)


def glycogen_production(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, MAX_BACTERIA_CAPACITY):
    hormone_level = (estrogen_level + progesterone_level) / 2
    total_bacteria_count = good_bacteria_count + bad_bacteria_count
    capacity_usage = total_bacteria_count / MAX_BACTERIA_CAPACITY

    if capacity_usage >= 1:
        glycogen_level = 0  # No glycogen left if we are at or over capacity
    elif capacity_usage > 0.7:
        # Rapid drop in glycogen as we approach capacity
        glycogen_level = random.uniform(
            0, 0.3) * (1 - capacity_usage) * hormone_level
    else:
        # Normal glycogen production but reduced as bacteria count grows
        glycogen_level = random.uniform(
            0.5, 1) * (1 - capacity_usage) * hormone_level

    return glycogen_level


def plotting(enviroment, microbe_trackers):

    # Plotting the results
    plt.figure(figsize=(12, 10))

    plt.subplot(7, 1, 1)
    plt.plot(range(1, len(enviroment["estrogen"]) + 1),
             enviroment["estrogen"], label='Estrogen Levels', linestyle='--')
    plt.plot(range(1, len(enviroment["progesterone"]) + 1),
             enviroment["progesterone"], label='Progesterone Levels', linestyle='--')
    plt.title('Hormone Levels Over Time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Hormone Levels')
    plt.legend()

    plt.subplot(7, 1, 2)
    plt.plot(range(1, len(enviroment["h2o2"]) + 1), enviroment["h2o2"],
             label='H2O2 Levels', linestyle='--', color='orange')
    plt.plot(range(1, len(enviroment["glycogen"]) + 1), enviroment["glycogen"],
             label='Gylcogen Level', linestyle='--', color='blue')
    plt.title('Chemical Levels over time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Level')
    plt.legend()

    plt.subplot(7, 1, 3)
    plt.plot(range(1, len(enviroment["iron"]) + 1), enviroment["iron"],
             label='Iron Levels', linestyle='--', color='orange')
    plt.title('Iron Pulse')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Iron Levels')
    plt.legend()

    plt.subplot(7, 1, 4)
    plt.plot(range(1, len(microbe_trackers["total bad"]) + 1),
             microbe_trackers["total bad"], label='Bad Bacteria', linestyle='--')
    plt.plot(range(1, len(microbe_trackers["total good"]) + 1),
             microbe_trackers["total good"], label='Good Bacteria', linestyle='--')
    plt.plot(range(1, len(microbe_trackers["total bacteria"]) + 1),
             microbe_trackers["total bacteria"], label='Total Bacteria', linestyle='--')
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
    for good_species in microbe_trackers["good bacteria step plot"]:
        plt.plot(range(1, len(microbe_trackers["good bacteria step plot"][good_species]) + 1),
                 microbe_trackers["good bacteria step plot"][good_species], label=f'{good_species} Count')
    plt.title('Good Bacteria Produced at step')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(7, 1, 6)
    for bad_species in microbe_trackers["bad bacteria step plot"]:
        plt.plot(range(1, len(microbe_trackers["bad bacteria step plot"][bad_species]) + 1),
                 microbe_trackers["bad bacteria step plot"][bad_species], label=f'{bad_species} Count')
    plt.title('Bad Bacteria Produced at step')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(7, 1, 7)
    for immune_type in microbe_trackers["immune tracker"]:
        plt.plot(range(1, len(microbe_trackers["immune tracker"][immune_type]) + 1),
                 microbe_trackers["immune tracker"][immune_type], label=f'{immune_type} Count')
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


# Function to initialize the current state for a specific step
def initialize_state(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count):
    state = {
        'estrogen_level': estrogen_level,
        'progesterone_level': progesterone_level,
        'iron_level': iron_pulse(estrogen_level, progesterone_level),
        'hydrogen_peroxide_level': 0.5,  # Initial default value
        'glycogen_level': glycogen_production(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, MAX_BACTERIA_CAPACITY)
    }
    return state

# Function to update the chemical levels in current_state


def update_chemical_levels(state, good_bacteria_count, bad_bacteria_count):
    state['iron_level'] = iron_pulse(
        state['estrogen_level'], state['progesterone_level'])
    state['glycogen_level'] = glycogen_production(
        state['estrogen_level'], state['progesterone_level'], good_bacteria_count, bad_bacteria_count, MAX_BACTERIA_CAPACITY)
    state['hydrogen_peroxide_level'] = h202_level(
        good_bacteria_count, bad_bacteria_count, state['estrogen_level'])

# Function to handle microbe interactions


def handle_microbes(microbes, state, microbe_trackers):
    for microbe in microbes:
        microbe.move()
        microbe.interact(state['estrogen_level'], state['progesterone_level'],
                         state['hydrogen_peroxide_level'], state['iron_level'], state['glycogen_level'])
        if microbe.species is None:
            continue
        elif "Good_Bacteria" in microbe.species:
            microbe_trackers["good bacteria step"][microbe.species] += 1
        elif "Bad_Bacteria" in microbe.species:
            microbe_trackers["bad bacteria step"][microbe.species] += 1

# Function to generate new immune cells


def generate_immune_cells(state, microbe_trackers):
    for _ in range(NUMBER_IMMUNE_CELLS):
        if state['estrogen_level'] > 0.5:
            target_type = np.random.choice(['kill_good', 'kill_bad'], p=[
                                           PROB_KILL_GOOD_MADE, PROB_KILL_BAD_MADE])
            new_immune_cell = Immune_Cell(
                target_type, (random.uniform(0, 10), random.uniform(0, 10)), 0)
            microbe_trackers['immune cells'][target_type].append(
                new_immune_cell)

# Function to handle immune cell interactions


def handle_immune_cells(microbe_trackers):
    for immune_type in microbe_trackers['immune cells']:
        for immune_cell in microbe_trackers['immune cells'][immune_type]:
            killed, good_bacteria_tracker, bad_bacteria_tracker = immune_cell.interact(
                microbe_trackers["good bacteria tracker"], microbe_trackers["bad bacteria tracker"])
            if killed:
                microbe_trackers['immune cells'][immune_type].remove(
                    immune_cell)

        # Track the number of immune cells for this type at the current step
        microbe_trackers["immune tracker"][immune_type].append(
            len(microbe_trackers['immune cells'][immune_type]))


# Function to append the current step counts to trackers
def update_trackers(microbe_trackers):
    for good_species in microbe_trackers["good bacteria tracker"]:
        microbe_trackers["good bacteria tracker"][good_species].append(
            microbe_trackers["good bacteria step"][good_species] +
            microbe_trackers["good bacteria tracker"][good_species][-1]
        )
    for bad_species in microbe_trackers["bad bacteria tracker"]:
        microbe_trackers["bad bacteria tracker"][bad_species].append(
            microbe_trackers["bad bacteria step"][bad_species] +
            microbe_trackers["bad bacteria tracker"][bad_species][-1]
        )

    for good_species in microbe_trackers["good bacteria step plot"]:
        microbe_trackers["good bacteria step plot"][good_species] =\
            [microbe_trackers["good bacteria tracker"][good_species][i+1] - microbe_trackers["good bacteria tracker"]
                [good_species][i] for i in range(len(microbe_trackers["good bacteria tracker"][good_species])-1)]
    for bad_species in microbe_trackers["bad bacteria step plot"]:
        microbe_trackers["bad bacteria step plot"][bad_species] =\
            [microbe_trackers["bad bacteria tracker"][bad_species][i+1] - microbe_trackers["bad bacteria tracker"]
                [bad_species][i] for i in range(len(microbe_trackers["bad bacteria tracker"][bad_species])-1)]

# Main simulation loop refactored


def simulation_loop():
    good_bacteria_proportion_tracker = []

    for test in range(TRIALS):
        print("Trial:", test + 1)

        # Read data
        data = read_data()
        estrogen_levels_raw = data[1]
        estrogen_levels = estrogen_levels_raw / max(estrogen_levels_raw)
        progesterone_levels_raw = data[2]
        progesterone_levels = progesterone_levels_raw / \
            max(progesterone_levels_raw)

        # Initialize environment dictionary to store hormone levels
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
            "immune cells": {'kill_good': [], 'kill_bad': []},
            "immune tracker": {'kill_good': [], 'kill_bad': []},
            "good bacteria step plot": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
            "bad bacteria step plot": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)}
        }

        total_good_bacteria = []
        total_bad_bacteria = []
        total_bacteria = []

        # Initialize persistent microbes
        microbes = [Microbe(species=f"Microbe_{i}", location=(random.uniform(
            0, 10), random.uniform(0, 10))) for i in range(NUM_MICROBES)]

        good_bacteria_count = 0
        bad_bacteria_count = 0

        for cycle in range(NUM_CYCLES):
            print(f"cycle {cycle}")
            for step in range(DAYS):
                # Get hormone levels for the current step
                estrogen_level = estrogen_levels[step]
                progesterone_level = progesterone_levels[step]

                # Initialize the state for the current step
                current_state = initialize_state(
                    estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count)

                # Store hormone levels
                enviroment["estrogen"].append(current_state['estrogen_level'])
                enviroment["progesterone"].append(
                    current_state['progesterone_level'])
                enviroment["iron"].append(current_state['iron_level'])
                enviroment["glycogen"].append(current_state['glycogen_level'])

                # Reset step trackers
                for good_species in microbe_trackers["good bacteria step"]:
                    microbe_trackers["good bacteria step"][good_species] = 0
                for bad_species in microbe_trackers["bad bacteria step"]:
                    microbe_trackers["bad bacteria step"][bad_species] = 0

                # Handle microbe interactions
                handle_microbes(microbes, current_state, microbe_trackers)

                # Generate new immune cells
                generate_immune_cells(current_state, microbe_trackers)

                # Handle immune cell interactions
                handle_immune_cells(microbe_trackers)

                # Update step trackers
                update_trackers(microbe_trackers)

                # Update chemical levels in the state
                update_chemical_levels(
                    current_state, good_bacteria_count, bad_bacteria_count)

                # Calculate total good, bad, and overall bacteria counts
                good_bacteria_count = sum(microbe_trackers["good bacteria tracker"][species][-1]
                                          for species in microbe_trackers["good bacteria tracker"])
                bad_bacteria_count = sum(microbe_trackers["bad bacteria tracker"][species][-1]
                                         for species in microbe_trackers["bad bacteria tracker"])
                total_bacteria_count = good_bacteria_count + bad_bacteria_count

                # Store total bacteria counts
                total_good_bacteria.append(good_bacteria_count)
                total_bad_bacteria.append(bad_bacteria_count)
                total_bacteria.append(total_bacteria_count)

                # Store hydrogen peroxide levels
                enviroment["h2o2"].append(
                    current_state['hydrogen_peroxide_level'])

        # Track good bacteria proportions at the end of the trial
        good_bacteria_proportion = good_bacteria_count / \
            (good_bacteria_count + bad_bacteria_count)
        good_bacteria_proportion_tracker.append(good_bacteria_proportion)

        # Add total bacteria to the microbe_trackers
        microbe_trackers["total good"] = total_good_bacteria
        microbe_trackers["total bad"] = total_bad_bacteria
        microbe_trackers["total bacteria"] = total_bacteria

    return enviroment, microbe_trackers, good_bacteria_proportion_tracker


enviroment, microbe_trackers, good_bacteria_proportion_tracker = simulation_loop()
plotting(enviroment, microbe_trackers)


# Calculate and print the average proportion of good bacteria across all trials
print(
    f"Average Proportion of good bacteria over {TRIALS} TRIALS is: {np.average(good_bacteria_proportion_tracker):.2f}")

# Calculate and print the standard deviation of the proportion of good bacteria across
# Calculate and print the standard deviation of the proportion of good bacteria across all trials
standard_deviation_proportion = np.std(good_bacteria_proportion_tracker)
print(
    f"The standard deviation of the trials is: {standard_deviation_proportion:.2f}")
