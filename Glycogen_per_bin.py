#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:23:31 2024

@author: matthewillsley
"""


import random
import matplotlib.pyplot as plt
import numpy as np

NUM_MICROBES = 50
DAYS = 28
NUM_CYCLES = 10
TRIALS = 1  # You can increase the number of trials
MAX_BACTERIA_CAPACITY = 100

GLYCOGEN_CAPASITY = 2  # The number of bacteria that a glycogen cluster can support
GLYCOGEN_INTERACTION_RANGE = 2

NUMBER_IMMUNE_CELLS = 10  # generated per step

# for both immune cells and glycogen consumption
IMMUNE_CELL_INTERACTION_RANGE = 3
NK_IMMUNE_CELL_INTERACTION_RANGE = 1*IMMUNE_CELL_INTERACTION_RANGE
IMMUNE_CELL_AGE_LIMIT = DAYS * 2
MACROPHAGE_KILL_LIMIT = 2
NEUTROPHIL_KILL_LIMIT = 1
NK_CELL_KILL_LIMIT = 1

SIDE_LENGTH = 100
grid_size = 100  # 100x100 grid
bin_size = 20  # Each bin is 10x10 units
num_bins = grid_size // bin_size

# Microbe Class


class Microbe:
    def __init__(self, species, location, is_new=False):
        self.species = species
        self.location = location
        self.alive = True  # Track whether the microbe is alive
        self.is_new = is_new  # Indicates if the microbe is newly created

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1))

        # Check if the new location is within bounds (0, SIDE_LENGTH)
        if new_location[0] < 0:
            new_location = (0, self.location[1])
        elif new_location[0] > SIDE_LENGTH:
            new_location = (SIDE_LENGTH, self.location[1])

        if new_location[1] < 0 or new_location[1] > SIDE_LENGTH:
            self.species = None  # Mark microbe as dead
            self.alive = False  # Mark it dead explicitly
            return True  # Microbe has died (out of bounds)

        self.location = new_location  # Update location if valid
        return False  # Microbe is still alive

    def interact_with_glycogen(self, glycogen_objects):
        glycogen_consumed = False
        # Check if any glycogen is close enough to be eaten
        for glycogen in glycogen_objects[:]:  # Iterate over a copy of the list
            if calculate_distance(self.location, glycogen.location) <= GLYCOGEN_INTERACTION_RANGE:
                # Bacteria "eats" the glycogen, reduce its amount
                if glycogen.reduce_amount(1/GLYCOGEN_CAPASITY):
                    # Remove the glycogen if it is depleted
                    glycogen_objects.remove(glycogen)
                glycogen_consumed = True

        if not glycogen_consumed:
            if self.is_new:
                # If the microbe is new and didn't find glycogen, it dies
                self.alive = False
                self.species = None  # Mark species as None to indicate death

        self.is_new = False
        return glycogen_consumed

    def interact(self, estrogen_level, progesterone_level, hydrogen_peroxide_level, iron_level, glycogen_level):
        if not self.alive:
            return  # Skip dead microbes

        hormone_level = (estrogen_level + progesterone_level) / 2

        # Microbe interaction logic based on environment conditions
        if hydrogen_peroxide_level >= 2 * iron_level and glycogen_level > 0:
            # Microbe changes to a good species if the environment is favorable
            self.species = random.choice(
                ["Good_Bacteria_1", "Good_Bacteria_2", "Good_Bacteria_3", "Good_Bacteria_4"])
        elif glycogen_level > 0:
            # If glycogen is available, but conditions are less favorable, microbe turns into a bad bacteria
            self.species = random.choice(["Bad_Bacteria_1", "Bad_Bacteria_2"])
        else:
            # If glycogen is depleted or conditions are poor, existing microbe hibernates
            if not self.is_new:
                print(
                    f"Existing microbe at {self.location} is hibernating due to poor environment.")
            else:
                # New microbes die instantly if conditions are poor
                self.alive = False
                self.species = None


class Glycogen:
    def __init__(self, location, amount=1.0):
        self.location = location  # (x, y) coordinates
        self.amount = amount  # Amount of glycogen in the clump

    def reduce_amount(self, amount):
        """Reduce the glycogen amount when bacteria consume it."""
        self.amount -= amount
        if self.amount <= 0:
            self.amount = 0  # Ensure the amount never goes below 0
            return True  # Return True if the glycogen is depleted
        return False  # Return False if there is still glycogen left


# Immune Cell Class


class ImmuneCell:
    def __init__(self, location, age=0):
        self.location = location
        self.age = age

    def move(self):
        # Basic random movement
        new_location = (self.location[0] + random.uniform(-1, 1),
                        self.location[1] + random.uniform(-1, 1))

        # Check if the new location is within bounds (0, 10)
        if new_location[0] < 0:
            new_location = (0, self.location[1])
        elif new_location[0] > SIDE_LENGTH:
            new_location = (SIDE_LENGTH, self.location[1])

        if new_location[1] < 0 or new_location[1] > SIDE_LENGTH:
            self.species = None  # Mark microbe as dead
            return True  # Microbe has died (out of bounds)

        self.location = new_location  # Update location if valid
        return False  # Microbe is still alive

    def age_cell(self):
        # General aging behavior for all immune cells
        self.age += 1
        if self.age > IMMUNE_CELL_AGE_LIMIT:

            return True  # Return True if the cell dies of old age
        return False


class Neutrophil(ImmuneCell):
    def __init__(self, location, age=0, kill_count=0):
        super().__init__(location, age)  # Initialize the base class (ImmuneCell)
        self.kill_count = kill_count

    def interact(self, good_bacteria_objects, bad_bacteria_objects, good_bacteria_tracker, bad_bacteria_tracker):
        # Macrophages can kill both good and bad bacteria in their proximity
        killed = False
        # Kill good bacteria if nearby
        for species in good_bacteria_objects.keys():
            # Iterate over a copy of the list
            for bacterium in good_bacteria_objects[species][:]:
                if calculate_distance(self.location, bacterium.location) <= IMMUNE_CELL_INTERACTION_RANGE:
                    good_bacteria_tracker[species][-1] -= 1
                    good_bacteria_objects[species].remove(bacterium)
                    self.kill_count += 1

                    killed = True
                    if self.kill_count >= NEUTROPHIL_KILL_LIMIT:  # Neutrophil dies after killing 2 bacteria
                        return True  # Return True if the macrophage dies

        # Kill bad bacteria if nearby
        for species in bad_bacteria_objects.keys():
            for bacterium in bad_bacteria_objects[species][:]:
                if calculate_distance(self.location, bacterium.location) <= IMMUNE_CELL_INTERACTION_RANGE:
                    bad_bacteria_tracker[species][-1] -= 1
                    bad_bacteria_objects[species].remove(bacterium)
                    self.kill_count += 1

                    killed = True
                    if self.kill_count >= NEUTROPHIL_KILL_LIMIT:  # Nuetrophil dies after killing 2 bacteria
                        return True

        return killed  # Return False if no bacteria were killed


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
                if calculate_distance(self.location, bacterium.location) <= IMMUNE_CELL_INTERACTION_RANGE:
                    good_bacteria_tracker[species][-1] -= 1
                    good_bacteria_objects[species].remove(bacterium)
                    self.kill_count += 1

                    killed = True
                    if self.kill_count >= MACROPHAGE_KILL_LIMIT:  # Macrophage dies after killing 3 bacteria
                        return True  # Return True if the macrophage dies

        # Kill bad bacteria if nearby
        for species in bad_bacteria_objects.keys():
            for bacterium in bad_bacteria_objects[species][:]:
                if calculate_distance(self.location, bacterium.location) <= IMMUNE_CELL_INTERACTION_RANGE:
                    bad_bacteria_tracker[species][-1] -= 1
                    bad_bacteria_objects[species].remove(bacterium)
                    self.kill_count += 1

                    killed = True
                    if self.kill_count >= MACROPHAGE_KILL_LIMIT:  # Macrophage dies after killing 3 bacteria
                        return True

        return killed  # Return False if no bacteria were killed


class NKcell(ImmuneCell):
    def __init__(self, location, age=0, kill_count=0):
        super().__init__(location, age)  # Initialize base class (ImmuneCell)
        self.kill_count = kill_count  # Track how many bacteria the NK cell has killed

    def interact(self, bad_bacteria_objects, bad_bacteria_tracker):
        # Natural Killer (NK) cells only kill bad bacteria
        killed = False

        # Kill bad bacteria if nearby
        for species in bad_bacteria_objects.keys():
            # Iterate over a copy of the list of bacteria
            for bacterium in bad_bacteria_objects[species][:]:
                # Check if bacteria is within interaction range
                if calculate_distance(self.location, bacterium.location) <= NK_IMMUNE_CELL_INTERACTION_RANGE:
                    # Reduce count in tracker
                    bad_bacteria_tracker[species][-1] -= 1
                    bad_bacteria_objects[species].remove(
                        bacterium)  # Remove bacteria
                    self.kill_count += 1
                    killed = True

                    # NK cells die after killing a certain number of bacteria
                    if self.kill_count >= NK_CELL_KILL_LIMIT:
                        return True  # NK cell dies after reaching the kill limit

        return killed  # Return True if a bacteria was killed, otherwise False


# Read data function
def read_data():
    hormone_data = np.genfromtxt("TabHormone.csv", comments='%',
                                 delimiter=",", skip_header=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    day_no = hormone_data[:, 0]
    est_level = hormone_data[:, 2]
    progest_level = hormone_data[:, 5]
    return day_no, est_level, progest_level


def initialize_state(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, glycogen_objects):
    # Produce glycogen and keep existing glycogen objects
    _, glycogen_objects = glycogen_production(
        estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, MAX_BACTERIA_CAPACITY, glycogen_objects)

    state = {
        'estrogen_level': estrogen_level,
        'progesterone_level': progesterone_level,
        'iron_level': iron_pulse(estrogen_level, progesterone_level),
        'hydrogen_peroxide_level': 0.5,  # Initial default value
        'glycogen_objects': glycogen_objects,  # Track the glycogen objects
        # Total number of glycogen clumps
        'glycogen_level': len(glycogen_objects),
        'cytokine_level': cytokine_level(good_bacteria_count, bad_bacteria_count)
    }
    return state

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


def cytokine_level(good_bacteria_count, bad_bacteria_count):
    total_bacteria = good_bacteria_count + bad_bacteria_count
    try:
        bad_bacteria_proportion = bad_bacteria_count/total_bacteria
        if bad_bacteria_proportion > 0.8:
            return random.uniform(0.8, 0.9)
        elif bad_bacteria_proportion > 0.4:
            return random.uniform(0.3, 0.7)
        else:
            return random.uniform(0, 0.3)
    except:
        return random.uniform(0, .2)


def glycogen_production(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, MAX_BACTERIA_CAPACITY, glycogen_objects):
    hormone_level = (estrogen_level + progesterone_level) / 2
    total_bacteria_count = good_bacteria_count + bad_bacteria_count
    capacity_usage = total_bacteria_count / MAX_BACTERIA_CAPACITY
    glycogen_count = 0

    # Avoid glycogen production if capacity is exceeded
    if capacity_usage >= 1:
        return glycogen_count, glycogen_objects

    try:
        # Calculate the number of glycogen clumps based on hormone levels and capacity usage
        clump_number = int(hormone_level * NUM_MICROBES * random.uniform(0, 1))
        for _ in range(clump_number):
            glycogen = Glycogen(location=(random.uniform(
                0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
            glycogen_objects.append(glycogen)
            glycogen_count += 1
    except Exception as e:
        print(f"Error in glycogen production: {e}")
        # Default to producing at least one glycogen object if something goes wrong
        glycogen = Glycogen(location=(random.uniform(
            0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
        glycogen_objects.append(glycogen)
        glycogen_count += 1

    return glycogen_count, glycogen_objects


# Function to update the chemical levels in current_state


def update_chemical_levels(state, good_bacteria_count, bad_bacteria_count):
    state['iron_level'] = iron_pulse(
        state['estrogen_level'], state['progesterone_level'])

    # Produce glycogen and update glycogen_objects
    _, state['glycogen_objects'] = glycogen_production(
        state['estrogen_level'], state['progesterone_level'], good_bacteria_count, bad_bacteria_count, MAX_BACTERIA_CAPACITY, state['glycogen_objects'])

    # Update glycogen_level based on the total number of clumps
    state['glycogen_level'] = len(state['glycogen_objects'])

    state['hydrogen_peroxide_level'] = h202_level(
        good_bacteria_count, bad_bacteria_count, state['estrogen_level'])
    state['cytokine_level'] = cytokine_level(
        good_bacteria_count, bad_bacteria_count)
    return state


# Function to handle microbe interactions


def handle_microbes(state, microbe_trackers, glycogen_objects):
    # Initialize new microbes, marking them as "new"
    microbes = [Microbe(species=f"Microbe_{i}", location=(random.uniform(
        0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)), is_new=True) for i in range(NUM_MICROBES)]

    for microbe in microbes:
        if microbe.move():  # Move the current microbe, and check if it died
            continue  # Skip dead microbes

        # Determine if microbe interacts with glycogen and consumes it
        microbe.interact_with_glycogen(glycogen_objects)

        # Handle microbe interactions (bacteria, hormone levels, etc.)
        microbe.interact(state['estrogen_level'], state['progesterone_level'],
                         state['hydrogen_peroxide_level'], state['iron_level'], state['glycogen_level'])

        if not microbe.alive or microbe.species is None:
            # Skip dead microbes
            continue

        elif "Good_Bacteria" in microbe.species:
            microbe_trackers["good object"][microbe.species].append(microbe)

        elif "Bad_Bacteria" in microbe.species:
            microbe_trackers["bad object"][microbe.species].append(microbe)

    # Remove dead microbes from the trackers
    for good_species in microbe_trackers["good object"]:
        microbe_trackers["good object"][good_species] = [
            m for m in microbe_trackers["good object"][good_species] if m.alive]

    for bad_species in microbe_trackers["bad object"]:
        microbe_trackers["bad object"][bad_species] = [
            m for m in microbe_trackers["bad object"][bad_species] if m.alive]


def generate_immune_cells(state, microbe_trackers):
    immune_cells_number = int(NUMBER_IMMUNE_CELLS *
                              (1+state['cytokine_level']))
    for _ in range(immune_cells_number):
        if state['estrogen_level'] > 0.1:
            immune_type = random.choice(["macrophage", "neutrophil", "NK"])
            if immune_type == "macrophage":
                new_immune_cell = Macrophage(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
            elif immune_type == "neutrophil":
                new_immune_cell = Neutrophil(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
            elif immune_type == "NK":
                new_immune_cell = NKcell(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))

            microbe_trackers['immune cells'].append(
                new_immune_cell)

# Function to handle immune cell interactions


def calculate_distance(location1, location2):
    return np.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)


def handle_immune_cells(microbe_trackers):
    for immune_cell in microbe_trackers['immune cells'][:]:

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
        elif isinstance(immune_cell, NKcell):
            killed = immune_cell.interact(microbe_trackers['bad object'],
                                          microbe_trackers["bad bacteria tracker"])

        # Remove immune cells if they die (either from killing or aging) or out of bounds
        if killed or immune_cell.age_cell() or immune_cell.move():
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


def get_bin(location):
    # Calculate the bin index, ensuring it's clamped within [0, num_bins - 1]
    x_bin = int(min(max(location[0] // bin_size, 0), num_bins - 1))
    y_bin = int(min(max(location[1] // bin_size, 0), num_bins - 1))
    return x_bin, y_bin


def create_heatmap(dictionary, entity_type="bacteria"):
    # Initialize an empty grid with the correct number of bins
    heatmap = np.zeros((num_bins, num_bins))

    if entity_type == "bacteria":
        # Loop over good bacteria
        for good_species in dictionary["good object"]:
            for bacterium in dictionary["good object"][good_species]:
                x_bin, y_bin = get_bin(bacterium.location)
                # Increment count in the correct bin
                heatmap[y_bin, x_bin] += 1

        # Loop over bad bacteria
        for bad_species in dictionary["bad object"]:
            for bacterium in dictionary["bad object"][bad_species]:
                x_bin, y_bin = get_bin(bacterium.location)
                heatmap[y_bin, x_bin] += 1

    elif entity_type == "immune_cells":
        # Loop over immune cells
        for immune_cell in dictionary['immune cells']:
            x_bin, y_bin = get_bin(immune_cell.location)
            heatmap[y_bin, x_bin] += 1

    elif entity_type == "glycogen":
        # Loop over immune cells
        for glycogen_clump in dictionary['glycogen_objects'][-1]:
            x_bin, y_bin = get_bin(glycogen_clump.location)
            heatmap[y_bin, x_bin] += 1

    return heatmap


def plot_heatmap(heatmap, title="Bacteria Distribution", cmap="hot"):
    immune_cell_circle_center = (grid_size/2, grid_size/2)
    glycogen_circle_centre = (2*grid_size/3, 2*grid_size/3)
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap=cmap, interpolation='nearest',
               origin='lower', extent=[0, grid_size, 0, grid_size])
    plt.colorbar(label='Count')
    circle = plt.Circle(immune_cell_circle_center, IMMUNE_CELL_INTERACTION_RANGE, color='blue',
                        fill=False, linewidth=2, linestyle='--', label="Immune Cell Interaction Range")
    plt.gca().add_patch(circle)  # Add the circle to the plot
    circle = plt.Circle(glycogen_circle_centre, GLYCOGEN_INTERACTION_RANGE, color='green',
                        fill=False, linewidth=2, linestyle='--', label=" Glycogen Interaction Range")
    plt.gca().add_patch(circle)
    plt.legend()
    plt.title(title)
    plt.xlabel('X Position ')
    plt.ylabel('Y Position')
    plt.show()


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

    plt.plot(range(1, len(enviroment["glycogen"]) + 1), enviroment["glycogen"],
             label='Gylcogen Level', linestyle='--', color='blue')

    plt.title('Total Glycogen Levels over time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Level')
    plt.legend()

    plt.subplot(7, 1, 3)
    plt.plot(range(1, len(enviroment["iron"]) + 1), enviroment["iron"],
             label='Iron Levels', linestyle='--', color='orange')
    plt.plot(range(1, len(enviroment["h2o2"]) + 1), enviroment["h2o2"],
             label='H2O2 Levels', linestyle='--', color='blue')
    plt.plot(range(1, len(enviroment["cytokine"]) + 1), enviroment["cytokine"],
             label='Cytokine level', linestyle='--', color='green')

    plt.title('Chemical levels')
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
        plt.plot(range(1, len(good_bacteria_tracker[good_species]) + 1),
                 good_bacteria_tracker[good_species], label=f'{good_species} Count')
    plt.title('Good Bacteria Produced Totals')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(7, 1, 6)
    for bad_species in bad_bacteria_tracker:
        plt.plot(range(1, len(bad_bacteria_tracker[bad_species]) + 1),
                 bad_bacteria_tracker[bad_species], label=f'{bad_species} Count')
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
    plt.savefig(fname="plots", dpi=1000)
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

        # Initialize environment dictionary to store hormone levels and glycogen objects
        environment = {
            "estrogen": [],
            "progesterone": [],
            "iron": [],
            "glycogen": [],
            "glycogen_objects": [],  # Track glycogen objects over time
            "h2o2": [],
            "cytokine": []
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

        # Initialize persistent microbes and glycogen objects
        good_bacteria_count = 0
        bad_bacteria_count = 0
        glycogen_objects = []  # Glycogen objects initialized

        for cycle in range(NUM_CYCLES):
            print(f"Cycle {cycle}")
            for step in range(DAYS):
                # Get hormone levels for the current step
                estrogen_level = estrogen_levels[step]
                progesterone_level = progesterone_levels[step]

                # Initialize the state for the current step, passing existing glycogen_objects
                current_state = initialize_state(
                    estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, glycogen_objects)

                # Store hormone levels and glycogen state in the environment
                environment["estrogen"].append(current_state['estrogen_level'])
                environment["progesterone"].append(
                    current_state['progesterone_level'])
                environment["iron"].append(current_state['iron_level'])
                environment["glycogen"].append(current_state['glycogen_level'])
                # Store a snapshot of the glycogen objects (copying the list)
                environment["glycogen_objects"].append(
                    [glycogen for glycogen in current_state['glycogen_objects']])
                environment["cytokine"].append(current_state['cytokine_level'])

                # Handle microbe interactions (including glycogen consumption)
                handle_microbes(
                    current_state, microbe_trackers, glycogen_objects)

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
                environment["h2o2"].append(
                    current_state['hydrogen_peroxide_level'])

        # Track good bacteria proportions at the end of the trial
        good_bacteria_proportion = good_bacteria_count / \
            (good_bacteria_count + bad_bacteria_count)
        good_bacteria_proportion_tracker.append(good_bacteria_proportion)

        # Add total bacteria to the microbe_trackers
        microbe_trackers["total good"] = total_good_bacteria
        microbe_trackers["total bad"] = total_bad_bacteria
        microbe_trackers["total bacteria"] = total_bacteria

    return environment, microbe_trackers, good_bacteria_proportion_tracker


def main():
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

    glycogen_heatmap = create_heatmap(
        enviroment, entity_type="glycogen")
    plot_heatmap(glycogen_heatmap,
                 title="Glycogen Distribution Heatmap", cmap="Greens")

    return enviroment, microbe_trackers, good_bacteria_proportion_tracker


enviroment, microbe_trackers, good_bacteria_proportion_tracker = main()
