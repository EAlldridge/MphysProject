#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:54:10 2024

@author: matthewillsley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:18:48 2024

@author: matthewillsley
"""


import random
import matplotlib.pyplot as plt
import numpy as np
NUM_MICROBES = 100
DAYS = 28
NUM_CYCLES = 15

TRIALS = 1  # You can increase the number of trials

PROB_KILL_GOOD_MADE = 0.4
PROB_KILL_BAD_MADE = 0.6

PROB_KILLS_BAD = 1  # 30% probability that good immune cells kill a bad bacteria
PROB_KILLS_GOOD = 1  # 90% probability that bad immune cells kill a good bacteria

MAX_BACTERIA_CAPACITY = 2000000

NUMBER_IMMUNE_CELLS = 20

IMMUNE_CELL_AGE_LIMIT = DAYS * 2
INTERACTION_RANGE = 1
SIDE_LENGTH = 100

grid_size = 10  # 100x100 grid
bin_size = 1  # Each bin is 10x10 units
num_bins = grid_size // bin_size

# Microbe Class


class Microbe:
    def __init__(self, species, location):
        self.species = species
        self.location = location

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1))

        # Check if the new location is within bounds (0, 10)
        if new_location[0] < 0:
            new_location = (0, self.location[1])
        elif new_location[0] > SIDE_LENGTH:
            new_location = (SIDE_LENGTH, self.location[1])

        if new_location[1] < 0:
            new_location = (self.location[0], 0)
        elif new_location[1] > SIDE_LENGTH:
            new_location = (self.location[0], SIDE_LENGTH)

        self.location = new_location  # Update location if valid
        return False  # Microbe is still alive

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


class ImmuneCell:
    def __init__(self, location, age=0):
        self.location = location
        self.age = age

    def move(self):
        # Basic random movement
        new_location = (self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1))

        # Ensure it stays within grid bounds (assuming SIDE_LENGTH is defined)
        self.location = (
            min(max(new_location[0], 0), SIDE_LENGTH),
            min(max(new_location[1], 0), SIDE_LENGTH)
        )
        return False  # Returning False means the immune cell is still alive

    def age_cell(self):
        # General aging behavior for all immune cells
        self.age += 1
        if self.age > IMMUNE_CELL_AGE_LIMIT:
            return True  # Return True if the cell dies of old age
        return False


class Neutrophil(ImmuneCell):
    def __init__(self, location, age=0):
        super().__init__(location, age)  # Initialize the base class (ImmuneCell)

    def interact(self, good_bacteria_objects, bad_bacteria_objects, good_bacteria_tracker, bad_bacteria_tracker):
        # Neutrophils interact with any bacteria (good or bad) in their proximity
        for species in good_bacteria_objects.keys():
            for bacterium in good_bacteria_objects[species]:
                if calculate_distance(self.location, bacterium.location) <= INTERACTION_RANGE:
                    # Kill one good bacterium
                    good_bacteria_tracker[species][-1] -= 1
                    good_bacteria_objects[species].remove(
                        bacterium)  # Remove from the objects list
                    print(
                        f"Neutrophil killed {species} (Good) at {self.location}")
                    return True  # Neutrophil dies after killing

        for species in bad_bacteria_objects.keys():
            for bacterium in bad_bacteria_objects[species]:
                if calculate_distance(self.location, bacterium.location) <= INTERACTION_RANGE:
                    # Kill one bad bacterium
                    bad_bacteria_tracker[species][-1] -= 1
                    bad_bacteria_objects[species].remove(
                        bacterium)  # Remove from the objects list
                    print(
                        f"Neutrophil killed {species} (Bad) at {self.location}")
                    return True  # Neutrophil dies after killing

        return False  # No interaction happened


class Macrophage(ImmuneCell):
    def __init__(self, location, age=0, kill_count=0):
        super().__init__(location, age)
        self.kill_count = kill_count  # Track how many bacteria the macrophage has killed

    def interact(self, good_bacteria_objects, bad_bacteria_objects, good_bacteria_tracker, bad_bacteria_tracker):
        # Macrophages can kill both good and bad bacteria in their proximity
        killed = False
        # Kill good bacteria if nearby
        for species in good_bacteria_objects.keys():
            # Iterate over a copy of the list
            for bacterium in good_bacteria_objects[species][:]:
                if calculate_distance(self.location, bacterium.location) <= INTERACTION_RANGE:
                    good_bacteria_tracker[species][-1] -= 1
                    good_bacteria_objects[species].remove(bacterium)
                    self.kill_count += 1
                    print(
                        f"Macrophage killed {species} (Good) at {self.location} (Total kills: {self.kill_count})")
                    killed = True
                    if self.kill_count >= 3:  # Macrophage dies after killing 3 bacteria
                        return True  # Return True if the macrophage dies

        # Kill bad bacteria if nearby
        for species in bad_bacteria_objects.keys():
            for bacterium in bad_bacteria_objects[species][:]:
                if calculate_distance(self.location, bacterium.location) <= INTERACTION_RANGE:
                    bad_bacteria_tracker[species][-1] -= 1
                    bad_bacteria_objects[species].remove(bacterium)
                    self.kill_count += 1
                    print(
                        f"Macrophage killed {species} (Bad) at {self.location} (Total kills: {self.kill_count})")
                    killed = True
                    if self.kill_count >= 3:  # Macrophage dies after killing 3 bacteria
                        return True

        return killed  # Return False if no bacteria were killed


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


def plotting(enviroment, microbe_trackers, good_bacteria_proportion_tracker):

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

    plt.plot(range(1, len(microbe_trackers["immune tracker"]) + 1),
             microbe_trackers["immune tracker"], label="immune cells")
    plt.title('Immune cells')
    plt.xlabel('Simulation Steps')
    plt.ylabel('# of immune cells')
    plt.legend()
    '''
    plt.subplot(7, 1, 7)
    plt.hist(good_bacteria_proportion_tracker, bins=5,
             density=False, label="Distribution of Trials")
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


def handle_microbes(state, microbe_trackers):
    microbes = [Microbe(species=f"Microbe_{i}", location=(random.uniform(
        0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH))) for i in range(NUM_MICROBES)]
    # Move all microbes at each time step
    for microbe in microbes:
        if microbe.move():  # Move the current microbe, and check if it died
            continue  # Skip dead microbes

        # Determine microbe interactions and their species change
        microbe.interact(state['estrogen_level'], state['progesterone_level'],
                         state['hydrogen_peroxide_level'], state['iron_level'], state['glycogen_level'])

        if microbe.species is None:
            continue  # Skip dead microbes

        elif "Good_Bacteria" in microbe.species:
            microbe_trackers["good object"][microbe.species].append(microbe)

        elif "Bad_Bacteria" in microbe.species:
            microbe_trackers["bad object"][microbe.species].append(microbe)

    # Move good bacteria and remove those that go out of bounds
    for good_species in list(microbe_trackers["good object"].keys()):
        for good_bacterium in microbe_trackers["good object"][good_species][:]:
            good_bacterium.move()  # Remove bacteria if it goes out of bounds

    # Move bad bacteria and remove those that go out of bounds
    for bad_species in list(microbe_trackers["bad object"].keys()):
        for bad_bacterium in microbe_trackers["bad object"][bad_species][:]:
            bad_bacterium.move()  # Remove bacteria if it goes out of bounds


# Function to generate new immune cells


def generate_immune_cells(state, microbe_trackers):
    for _ in range(NUMBER_IMMUNE_CELLS):
        if state['estrogen_level'] > 0.5:
            immune_type = random.choice(["macrophage", "neutrophil"])
            if immune_type == "macrophage":
                new_immune_cell = Macrophage(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
            elif immune_type == "neutrophil":
                new_immune_cell = Neutrophil(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))

            microbe_trackers['immune cells'].append(
                new_immune_cell)

# Function to handle immune cell interactions


def calculate_distance(location1, location2):
    return np.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)


def handle_immune_cells(microbe_trackers):
    for immune_cell in microbe_trackers['immune cells'][:]:
        if immune_cell.move():  # Check if the immune cell moves out of bounds
            microbe_trackers['immune cells'].remove(immune_cell)
            continue

        # Interact with both good and bad bacteria
        killed = False
        if isinstance(immune_cell, Neutrophil):
            # Neutrophils interact with any bacteria (good or bad)
            killed = immune_cell.interact(microbe_trackers["good object"],
                                          microbe_trackers["bad object"],
                                          microbe_trackers["good bacteria tracker"],
                                          microbe_trackers["bad bacteria tracker"])
        elif isinstance(immune_cell, Macrophage):
            # Macrophages interact with any bacteria (good or bad)
            killed = immune_cell.interact(microbe_trackers["good object"],
                                          microbe_trackers["bad object"],
                                          microbe_trackers["good bacteria tracker"],
                                          microbe_trackers["bad bacteria tracker"])

        # Remove immune cells if they die (either from killing or aging)
        if killed or immune_cell.age_cell():
            microbe_trackers['immune cells'].remove(immune_cell)
    microbe_trackers['immune tracker'].append(
        len(microbe_trackers['immune cells']))


# Function to append the current step counts to trackers
def update_trackers(microbe_trackers):
    # Update good bacteria tracker based on the lengths of the good object lists
    for good_species in microbe_trackers["good bacteria tracker"]:
        good_bacteria_count = len(
            microbe_trackers["good object"][good_species])
        microbe_trackers["good bacteria tracker"][good_species].append(
            good_bacteria_count)

    # Update bad bacteria tracker based on the lengths of the bad object lists
    for bad_species in microbe_trackers["bad bacteria tracker"]:
        bad_bacteria_count = len(microbe_trackers["bad object"][bad_species])
        microbe_trackers["bad bacteria tracker"][bad_species].append(
            bad_bacteria_count)

    # Calculate step-based plot data for good bacteria
    for good_species in microbe_trackers["good bacteria step plot"]:
        microbe_trackers["good bacteria step plot"][good_species] = [
            microbe_trackers["good bacteria tracker"][good_species][i+1] -
            microbe_trackers["good bacteria tracker"][good_species][i]
            for i in range(len(microbe_trackers["good bacteria tracker"][good_species]) - 1)
        ]

    # Calculate step-based plot data for bad bacteria
    for bad_species in microbe_trackers["bad bacteria step plot"]:
        microbe_trackers["bad bacteria step plot"][bad_species] = [
            microbe_trackers["bad bacteria tracker"][bad_species][i+1] -
            microbe_trackers["bad bacteria tracker"][bad_species][i]
            for i in range(len(microbe_trackers["bad bacteria tracker"][bad_species]) - 1)
        ]


grid_size = 100  # Example: 100x100 grid
bin_size = 10    # Each bin is 10x10 units
num_bins = grid_size // bin_size  # Number of bins along each axis


def get_bin(location):
    # Calculate the bin index, ensuring it's clamped within [0, num_bins - 1]
    x_bin = int(min(max(location[0] // bin_size, 0), num_bins - 1))
    y_bin = int(min(max(location[1] // bin_size, 0), num_bins - 1))
    return x_bin, y_bin


def create_heatmap(microbe_trackers, entity_type="bacteria"):
    # Initialize an empty grid with the correct number of bins
    heatmap = np.zeros((num_bins, num_bins))

    if entity_type == "bacteria":
        # Loop over good bacteria
        for good_species in microbe_trackers["good object"]:
            for bacterium in microbe_trackers["good object"][good_species]:
                x_bin, y_bin = get_bin(bacterium.location)
                # Increment count in the correct bin
                heatmap[y_bin, x_bin] += 1

        # Loop over bad bacteria
        for bad_species in microbe_trackers["bad object"]:
            for bacterium in microbe_trackers["bad object"][bad_species]:
                x_bin, y_bin = get_bin(bacterium.location)
                heatmap[y_bin, x_bin] += 1

    elif entity_type == "immune_cells":
        # Loop over immune cells
        for immune_cell in microbe_trackers['immune cells']:
            x_bin, y_bin = get_bin(immune_cell.location)
            heatmap[y_bin, x_bin] += 1

    return heatmap


def plot_heatmap(heatmap, title="Bacteria Distribution", cmap="hot"):
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap=cmap, interpolation='nearest',
               origin='lower', extent=[0, grid_size, 0, grid_size])
    plt.colorbar(label='Count')
    plt.title(title)
    plt.xlabel('X Position ')
    plt.ylabel('Y Position')
    plt.show()


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
            "immune cells": [],
            "immune tracker": [],
            "good bacteria step plot": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
            "bad bacteria step plot": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)},
            "good object": {f"Good_Bacteria_{i}": [] for i in range(1, 5)},
            "bad object": {f"Bad_Bacteria_{i}": [] for i in range(1, 3)}
        }

        total_good_bacteria = []
        total_bad_bacteria = []
        total_bacteria = []

        # Initialize persistent microbes

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

                handle_microbes(current_state, microbe_trackers)

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
plotting(enviroment, microbe_trackers, good_bacteria_proportion_tracker)


# Calculate and print the average proportion of good bacteria across all trials
print(
    f"Average Proportion of good bacteria over {TRIALS} TRIALS is: {np.average(good_bacteria_proportion_tracker):.2f}")

# Calculate and print the standard deviation of the proportion of good bacteria across
# Calculate and print the standard deviation of the proportion of good bacteria across all trials
standard_deviation_proportion = np.std(good_bacteria_proportion_tracker)
print(
    f"The standard deviation of the trials is: {standard_deviation_proportion:.2f}")

# Create heatmap for bacteria
bacteria_heatmap = create_heatmap(microbe_trackers, entity_type="bacteria")
plot_heatmap(bacteria_heatmap,
             title="Bacteria Distribution Heatmap", cmap="Reds")

# Create heatmap for immune cells
immune_cell_heatmap = create_heatmap(
    microbe_trackers, entity_type="immune_cells")
plot_heatmap(immune_cell_heatmap,
             title="Immune Cell Distribution Heatmap", cmap="Blues")
