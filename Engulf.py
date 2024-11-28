#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:23:31 2024

@author: matthewillsley
"""


import random
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

DAYS = 28
NUM_CYCLES = 10
TRIALS = 10 # You can increase the number of trials
MAX_BACTERIA_CAPACITY = 10000
HIBERNATION_TIME = DAYS * NUM_CYCLES
GLYCOGEN_SCALE = 100  # maximum number of new glycogen clumps generated per step
GLYCOGEN_CAPACITY = 6  # The number of bacteria that a glycogen cluster can support
GLYCOGEN_INTERACTION_RANGE = 1
REPLICATE_PROB = 0.5
# generated per step (ish) can be more if cytokines are high
NUMBER_IMMUNE_CELLS = 3
INITIAL_GOOD = 100
INITIAL_BAD = 200
INITIAL_CA = 0
JENSII = True #if want one species to dominate with CA125

# for both immune cells and glycogen consumption
IMMUNE_CELL_INTERACTION_RANGE = 0.5
PROB_KILL = 0.2
NK_IMMUNE_CELL_INTERACTION_RANGE = 2 * IMMUNE_CELL_INTERACTION_RANGE
IMMUNE_CELL_AGE_LIMIT = DAYS * 2
MACROPHAGE_KILL_LIMIT = 3
NEUTROPHIL_KILL_LIMIT = 2
NK_CELL_KILL_LIMIT = 2

HALF_LIFE = 20

SIDE_LENGTH = 10
grid_size = 10  # 10x10 grid
bin_size = 2  # Each bin is 10x10 units
num_bins = grid_size // bin_size

# Microbe Class


class Microbe:
    def __init__(self, species, location, time_without_glycogen=0):
        self.species = species
        self.location = location
        self.alive = True  # Track whether the microbe is alive
        # Indicates if the microbe is newly created
        self.time_without_glycogen = time_without_glycogen

    def move(self):
        new_location = (self.location[0] + random.uniform(-SIDE_LENGTH * 0.1, SIDE_LENGTH * 0.1),
                        self.location[1] + random.uniform(-SIDE_LENGTH * 0.1, SIDE_LENGTH * 0.1))

        # Check if the new location is within bounds (0, SIDE_LENGTH)
        if new_location[0] < 0:
            new_location = (0, self.location[1])
        elif new_location[0] > SIDE_LENGTH:
            new_location = (SIDE_LENGTH, self.location[1])

        if new_location[1] < 0 or new_location[1] > SIDE_LENGTH:
            self.alive = False  # Mark microbe as dead
            return True  # Microbe has died (out of bounds)

        self.location = new_location  # Update location if valid
        return False  # Microbe is still alive

    def interact_with_glycogen(self, glycogen_objects):
        glycogen_consumed = False
        for glycogen in glycogen_objects[:]:  # Iterate over a copy of the list
            if calculate_distance(self.location, glycogen.location) <= GLYCOGEN_INTERACTION_RANGE:
                self.time_without_glycogen = 0  # Reset counter on glycogen consumption
                glycogen_consumed = True
                if glycogen.reduce_amount(1 / GLYCOGEN_CAPACITY):
                    # Remove depleted glycogen
                    glycogen_objects.remove(glycogen)
                break  # Stop looking once glycogen is consumed

        if not glycogen_consumed:
            self.time_without_glycogen += 1

        # Kill microbe if it has been without glycogen for too long
        if self.time_without_glycogen > HIBERNATION_TIME:
            self.alive = False

            return False
        return glycogen_consumed

    def replicate(self, glycogen_objects, CA125_objects):
        if not self.alive:
            return None  # Dead microbes don't replicate

        nearby_glycogen = 0
        for glycogen in glycogen_objects:
            if calculate_distance(glycogen.location, self.location) <= GLYCOGEN_INTERACTION_RANGE:
                nearby_glycogen += glycogen.amount

        # Adjust replication condition: More bacteria are made if there is some glycogen nearby
        # Make this threshold low enough to allow for more replication
        if nearby_glycogen > 1 and "Good_Bacteria" in self.species and random.uniform(0, 1) < REPLICATE_PROB:
            if JENSII == True and self.species == "Good_Bacteria_1":
                new_bacterium = Microbe(self.species, self.location)
                return new_bacterium
            else:                
                if calculate_local_concentration(self.location, CA125_objects) < 4:
                    new_bacterium = Microbe(self.species, self.location)
                    return new_bacterium
                
        elif nearby_glycogen > 1 and "Bad_Bacteria" in self.species and random.uniform(0, 1) < REPLICATE_PROB:
            new_bacterium = Microbe(self.species, self.location)
            return new_bacterium

        else:

            return None  # No replication if no glycogen is nearby
        
    def interact_with_bacteriocins(self, bacteriocin_objects, bacteria):
        bacteriocin = next((x for x in bacteriocin_objects if calculate_distance(x.location, self.location) <= 0.1), None)
        if bacteriocin != None:
            bacteria.remove(self)
            bacteriocin_objects.remove(bacteriocin)
            


class Inflammation:
    def __init__(self, location, measure = 0):
        self.location = location
        self.measure = measure
        
    def inflammation_calc(self, CA125, cytokines):
        for obj in CA125:
            if self.location[0] <= obj.location[0] <= self.location[0]+1 and self.location[1] <= obj.location[1] <= (self.location[1]+1):
                self.measure += 0.5
                
    def cytokine_increase(self, cytokine_object):
        if self.measure > 3:
            t = random.randint(1,2)
            for i in range(t):
                cytokine_object.append(Protein("Cytokine", self.location))
            return t, cytokine_object
        
        if self.measure > 10:
            t = random.randint(2,3)
            for i in range(t):
                cytokine_object.append(Protein("Cytokine", self.location))
            return t, cytokine_object
            
        else:
            return 0, cytokine_object
            


class Protein:
    def __init__(self, species, location, alive = True):
        self.species = species
        self.location = location
        self.alive = True
    
    def move(self, CA_objects):
        # Define small movement step size
        step_size = SIDE_LENGTH * 0.2

        # Check concentration in adjacent directions
        directions = [
            (step_size, 0),  # Right
            (-step_size, 0),  # Left
            (0, step_size),  # Up
            (0, -step_size)  # Down
        ]

        min_concentration = float('inf')
        best_direction = (0, 0)

        for dx, dy in directions:
            new_location = (self.location[0] + dx, self.location[1] + dy)
            if 0 <= new_location[0] <= SIDE_LENGTH and 0 <= new_location[1] <= SIDE_LENGTH:
                concentration = calculate_local_concentration(
                    new_location, CA_objects)
                if concentration < min_concentration:
                    min_concentration = concentration
                    best_direction = (dx, dy)

        # Move in the best direction with the lowest concentration
        self.location = (
            self.location[0] + best_direction[0], self.location[1] + best_direction[1])

        # Boundary check to ensure it stays within valid space
        if self.location[0] < 0:
            self.location = (0, self.location[1])
        elif self.location[0] > SIDE_LENGTH:
            self.location = (SIDE_LENGTH, self.location[1])

        if self.location[1] < 0:
            self.alive = False  # Mark as dead if out of bounds
            return True  # moved out of bounds and is considered "dead"

        return False
   

class Glycogen:
    def __init__(self, location, amount=1.0, alive=True):
        self.location = location  # (x, y) coordinates
        self.amount = amount  # Amount of glycogen in the clump
        self.alive = True

    def reduce_amount(self, amount):
        """Reduce the glycogen amount when bacteria consume it."""
        self.amount -= amount
        if self.amount <= 0:
            self.amount = 0  # Ensure the amount never goes below 0
            return True  # Return True if the glycogen is depleted
        return False  # Return False if there is still glycogen left

    def move(self, glycogen_objects):
        # Define small movement step size
        step_size = SIDE_LENGTH * 0.2

        # Check concentration in adjacent directions
        directions = [
            (step_size, 0),  # Right
            (-step_size, 0),  # Left
            (0, step_size),  # Up
            (0, -step_size)  # Down
        ]

        min_concentration = float('inf')
        best_direction = (0, 0)

        for dx, dy in directions:
            new_location = (self.location[0] + dx, self.location[1] + dy)
            if 0 <= new_location[0] <= SIDE_LENGTH and 0 <= new_location[1] <= SIDE_LENGTH:
                concentration = calculate_local_concentration(
                    new_location, glycogen_objects)
                if concentration < min_concentration:
                    min_concentration = concentration
                    best_direction = (dx, dy)

        # Move in the best direction with the lowest concentration
        self.location = (
            self.location[0] + best_direction[0], self.location[1] + best_direction[1])

        # Boundary check to ensure it stays within valid space
        if self.location[0] < 0:
            self.location = (0, self.location[1])
        elif self.location[0] > SIDE_LENGTH:
            self.location = (SIDE_LENGTH, self.location[1])

        if self.location[1] < 0 or self.location[1] > SIDE_LENGTH:
            self.alive = False  # Mark as dead if out of bounds
            return True  # Glycogen has moved out of bounds and is considered "dead"

        return False  # Glycogen is still alive

# Immune Cell Class


class ImmuneCell:
    def __init__(self, location, age=0):
        self.location = location
        self.age = age

    def move(self, bacteria):
        # Define small movement step size
        step_size = SIDE_LENGTH * 0.2

        # Check concentration in adjacent directions
        directions = [
            (step_size, 0),  # Right
            (-step_size, 0),  # Left
            (0, step_size),  # Up
            (0, -step_size)  # Down
        ]

        min_concentration = float('0')
        best_direction = (0, 0)

        for dx, dy in directions:
            new_location = (self.location[0] + dx, self.location[1] + dy)
            if 0 <= new_location[0] <= SIDE_LENGTH and 0 <= new_location[1] <= SIDE_LENGTH:
                concentration = calculate_local_concentration(
                    new_location, bacteria)
                if concentration > min_concentration:
                    min_concentration = concentration
                    best_direction = (dx, dy)

        # Move in the best direction with the lowest concentration
        self.location = (
            self.location[0] + best_direction[0], self.location[1] + best_direction[1])

        # Boundary check to ensure it stays within valid space
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

    def interact(self, all_bacteria):
        # Macrophages can kill both good and bad bacteria in their proximity
        killed = False
        # Kill good bacteria if nearby

        for bacterium in all_bacteria:
            if calculate_distance(self.location, bacterium.location) <= IMMUNE_CELL_INTERACTION_RANGE\
                    and PROB_KILL < random.uniform(0, 1):
                bacterium.alive = False
                all_bacteria.remove(bacterium)
                self.kill_count += 1

                if self.kill_count >= NEUTROPHIL_KILL_LIMIT:
                    killed = True
        return killed, all_bacteria


class Macrophage(ImmuneCell):
    def __init__(self, location, age=0, kill_count=0):
        super().__init__(location, age)
        self.kill_count = kill_count  # Track how many bacteria the macrophage has killed

    def interact(self, all_bacteria):
        killed = False
        for bacterium in all_bacteria:
            if calculate_distance(self.location, bacterium.location) <= IMMUNE_CELL_INTERACTION_RANGE \
                    and PROB_KILL < random.uniform(0, 1):
                bacterium.alive = False
                all_bacteria.remove(bacterium)
                self.kill_count += 1
                #engulf if close bacteria
                another_bacterium = next((x for x in all_bacteria if calculate_distance(x.location, bacterium.location) <= 0.01), None)
                if another_bacterium != None and another_bacterium != bacterium:
                    another_bacterium.alive = False
                    all_bacteria.remove(another_bacterium)
                    print("Engulfed!")
                    self.kill_count += 1
                    
                if self.kill_count >= MACROPHAGE_KILL_LIMIT:
                    killed = True
        return killed, all_bacteria


class NKcell(ImmuneCell):
    def __init__(self, location, age=0, kill_count=0):
        super().__init__(location, age)  # Initialize base class (ImmuneCell)
        self.kill_count = kill_count  # Track how many bacteria the NK cell has killed

    def interact(self, all_bacteria):
        killed = False
        for bacterium in all_bacteria:
            if "Bad_Bacteria" in bacterium.species:
                if calculate_distance(self.location, bacterium.location) <= NK_IMMUNE_CELL_INTERACTION_RANGE \
                        and PROB_KILL < random.uniform(0, 1):
                    bacterium.alive = False
                    all_bacteria.remove(bacterium)
                    self.kill_count += 1

                    if self.kill_count >= NK_CELL_KILL_LIMIT:
                        killed = True
        return killed, all_bacteria


class Tumour():
    def __init__(self, stage, grade, size, age, asymptomatic, speed, initial):
        self.stage = stage
        self.grade = grade
        self.size = size
        self.age = age
        self.asymptomatic = asymptomatic
        self.speed = speed
        self.initial = initial
    
    def secretions(self):
        add = 0
        self.age += 1
        
        if self.size > 100:
            add = random.uniform(0.6,1)
        elif self.size > 10:
            add = random.uniform(0.01,0.1)
        if self.stage == 1:
            if self.asymptomatic == False: #50% low grade have elevated
                return random.uniform(0.2,0.6) + add
            else:
                return 0.1
            #less
        elif self.stage == 2:
            #more
            if self.asymptomatic == False:
                return random.uniform(0.3,0.6)+add
            else:
                return 0.1
            
        elif self.stage > 2:
            if self.asymptomatic == False: #80%
                return random.uniform(0.6,1)+add
            else:
                return 0.1
        else:
            return 0

    def growth(self, VEGF):
        if self.stage != 0:
            if self.speed == "Slow":
                alpha = 523000
                beta = 5
                gamma = 0.005
            if self.speed == "Fast":
                alpha = 523000
                beta = 5
                gamma = 0.01
            new_size = alpha*np.exp(-beta*np.exp(-gamma*self.age))
            self.size = new_size
            if self.size == 1:
                self.initial[2] = new_size #accurate initial size to age
            
        if self.size > 9000:
            self.grade = 4
        if self.size > 6000:
            self.grade = 3
        if self.size > 3000:
            self.grade = 2
        
        if VEGF > 50 and self.stage < 1:
            self.stage = 1
        if VEGF > 100 and self.stage < 2:
            self.stage = 2
        if VEGF > 200 and self.stage < 3:
            self.stage = 3
        if VEGF > 500 and self.stage < 4:
            self.stage = 4
    

# Read data function
def read_data():
    hormone_data = np.genfromtxt("TabHormone.csv", comments='%',
                                 delimiter=",", skip_header=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    day_no = hormone_data[:, 0]
    est_level = hormone_data[:, 2]
    progest_level = hormone_data[:, 5]
    return day_no, est_level, progest_level

def generate_inflammation(inflammation_objects, CA125_objects, cytokines):
    for cell in inflammation_objects:
        cell.measure = 0 #remove this if want inflammation to be continuous
        Inflammation.inflammation_calc(cell, CA125_objects, cytokines)
    return inflammation_objects

def Inflammatory_response(microbe_trackers, cytokine_objects):
    for cytokine in cytokine_objects:
        generate_immune_cells(None, microbe_trackers, cytokine.location)

def initialize_state(estrogen_level, progesterone_level, good_bacteria_count
                     , bad_bacteria_count, glycogen_objects, step, pH_level, 
                     tumour, CA125_objects, inflammation_objects, cycle):
    # Produce glycogen and keep existing glycogen objects
    glycogen_objects = glycogen_production(
        estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, MAX_BACTERIA_CAPACITY, glycogen_objects)
    CA125_objects = CA_calc(estrogen_level, progesterone_level, tumour, CA125_objects, step, cycle)
    cytokines = cytokine_level(good_bacteria_count, bad_bacteria_count)
    inflammation_objects = generate_inflammation(inflammation_objects, CA125_objects, cytokines)
    
    state = {
        'estrogen_level': estrogen_level,
        'progesterone_level': progesterone_level,
        'iron_level': iron_pulse(estrogen_level, progesterone_level),
        # Initial default value
        'pH': pH(pH_level, good_bacteria_count, bad_bacteria_count, step),
        'glycogen_objects': glycogen_objects,  # Track the glycogen objects
        # Total number of glycogen clumps
        'glycogen_level': len(glycogen_objects),
        'cytokine_level': cytokines,
        "CA125_objects": CA125_objects,
        "CA125_level": len(CA125_objects),
        "Inflammation": inflammation_objects,
        "VEGF_level": VEGF_calc(tumour, step)
    }
    return state

# Iron pulse function


def iron_pulse(estrogen_level, progesterone_level):
    if estrogen_level < 0.2 and progesterone_level < 0.2:
        iron_level = random.uniform(0.7, 1)
    else:
        iron_level = random.uniform(0.1, 0.3)
    return iron_level


def solution_pH_calc(pH1, pH2, pH3, vol1, vol2, vol3):
    M1 = 10**-pH1
    # print(pH2)
    M2 = 10**-pH2
    M3 = 10**-pH3
    mol1 = M1*vol1
    mol2 = M2*vol2
    mol3 = M3*vol3
    mol = mol1 + mol2 + mol3
    M = mol / (vol1+vol2+vol3)
    pH = -np.log10(M)
    return pH


def pH(pH, total_good_bacteria, total_bad_bacteria, step):
    # during menstruation menstrual blood reduces
   # print(environment["pH"])
    if step < 7:
        # pH blood
        pH_b = random.uniform(7.3, 7.5)
        if len(pH) == 0:
            # pH vagina
            pH_v = random.uniform(3.5, 4.5)
            vol_s = 0.05*10**-3

        else:
            total_bacteria = total_good_bacteria + total_bad_bacteria
            bacteria_proportion = total_good_bacteria/total_bacteria
            # initial pH
            pH_v = pH[-1]
            vol_s = 0.5*bacteria_proportion*10**-3

        # pH of secretion and volumes
        pH_s = random.uniform(3.5, 4.5)
        vol_b = 1*10**-3
        vol_v = 0.4*10**-3

    else:
        total_bacteria = total_good_bacteria + total_bad_bacteria
        bacteria_proportion = total_good_bacteria/total_bacteria
        pH_v = pH[-1]
        pH_s = random.uniform(3.5, 4.5)
        vol_v = 1*10**-3
        vol_s = 0.5*bacteria_proportion*10**-3

        pH_b = 0
        vol_b = 0

    new_pH = solution_pH_calc(pH_b, pH_v, pH_s, vol_b, vol_v, vol_s)
    new_pH = new_pH + random.uniform(-0.2, 0.2)
    return new_pH


def cytokine_level(good_bacteria_count, bad_bacteria_count):
    total_bacteria = good_bacteria_count + bad_bacteria_count
    try:
        bad_bacteria_proportion = bad_bacteria_count/total_bacteria
        if bad_bacteria_proportion > 0.8 or bad_bacteria_proportion < 0.2:
            return random.uniform(0.8, 0.9)
        elif bad_bacteria_proportion > 0.5:
            return random.uniform(0.3, 0.7)
        else:
            return random.uniform(0, 0.3)
    except:
        return random.uniform(0, .2)


def calculate_local_concentration(location, cell_objects, radius=1):
    count = 0
    for cell in cell_objects:
        if cell != location and calculate_distance(location, cell.location) <= radius:
            count += 1
    return count

def glycogen_production(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, MAX_BACTERIA_CAPACITY, glycogen_objects):
    hormone_level = (estrogen_level + progesterone_level) / 2
    total_bacteria_count = good_bacteria_count + bad_bacteria_count
    capacity_usage = total_bacteria_count / MAX_BACTERIA_CAPACITY
    glycogen_count = 0

    # Avoid glycogen production if capacity is exceeded
    if capacity_usage >= 1:
        return glycogen_objects

    try:
        # Calculate the number of glycogen clumps based on hormone levels and capacity usage
        clump_number = int(hormone_level * GLYCOGEN_SCALE)
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

    return glycogen_objects


def move_glycogen(glycogen_objects):
    for glycogen in glycogen_objects:
        glycogen.move(glycogen_objects)
        if glycogen.alive == False:
            glycogen_objects.remove(glycogen)


def protein_diffusion(protein_objects):
    for protein in protein_objects:
        protein.move(protein_objects)
        if protein.alive == False:
            protein_objects.remove(protein)

def natural_CA125(estrogen, progesterone):
    if estrogen < 0.3 and progesterone < 0.3:
        return random.uniform(0,0.5)
    elif estrogen > 0.5:
        return random.uniform(0.5,1.2)
    else:
        return random.uniform(0,1)  


def CA_calc(estrogen, progesterone, tumour, CA125_objects, step, cycle):
    CA_increase = natural_CA125(estrogen, progesterone)
    CA_increase += Tumour.secretions(tumour)
    current_CA = (INITIAL_CA - CA_increase/(np.log(2)/HALF_LIFE))*np.exp(-(np.log(2)/HALF_LIFE)*(28*cycle + step)) + CA_increase/(np.log(2)/HALF_LIFE)
    #print(current_CA)
    #print(len(CA125_objects))
    previous = len(CA125_objects)
    additional = current_CA - len(CA125_objects)
    additional = int(additional)
    #print(additional)
    if additional > 0:
        for _ in range(additional):
            CA = Protein("CA125", (random.randint(0,SIDE_LENGTH), random.randint(SIDE_LENGTH-1,SIDE_LENGTH)))#TOP TWO ROWS
            CA125_objects.append(CA)   
    
    else:
        #print(CA125_objects)
        for _ in range (-(additional)):
            X = random.randint(0,previous-1)
            CA125_objects.remove(CA125_objects[X])
            previous -= 1
    protein_diffusion(CA125_objects)
            
    return CA125_objects

def VEGF_calc(tumour, step):
    additional = 0
    if tumour.size < 8: #little effect here since tumour small& sufficient blood
        return 0
    if step < 7:
        additional = 1
    elif tumour.stage == 0:
        VEGF = 0
    VEGF = tumour.stage + additional
    return VEGF
    
    
def generate_tumour():
    stage = random.randint(0,4)
    stage = 4
    asymptomatic = True   
    
    if stage == 1:
        grade = random.randint(1,2)
        age = random.randint(0,2)
        if random.randint(1,2) == 2: #50%
            asymptomatic = False
        
    elif stage == 2:
        grade = random.randint(1,3)
        age = random.randint(2,5)
        if random.randint(1, 7) == 1:
            asymptomatic = False      
        
    elif stage == 3:
        grade = random.randint(2,3)
        age = random.randint(5,8)
        if random.randint(1, 7) == 1:
            asymptomatic = False 
         
    elif stage == 4:
        grade = random.randint(2,3)
        age = random.randint(8,12)
        if random.randint(1, 5) == 1: #80%
            asymptomatic = False
        
    else:
        grade = 0
        age = 0
    
    speed = random.choice(["Fast", "Slow"])
    initial_tumour = [stage, grade, 1, age]
    speed = "Fast"
    asymptomatic = False
    tumour = Tumour(stage, grade, 1, age, asymptomatic, speed, initial = initial_tumour)
    return tumour
# Function to update the chemical levels in current_state


def update_chemical_levels(environment, state, good_bacteria_count, bad_bacteria_count, step, cytokine_objects):
    state['iron_level'] = iron_pulse(
        state['estrogen_level'], state['progesterone_level'])

    # Produce glycogen and update glycogen_objects

    # Update glycogen_level based on the total number of clumps
    state['glycogen_level'] = len(state['glycogen_objects'])

    state['pH'] = environment["pH"]
    state['cytokine_level'] = cytokine_level(
        good_bacteria_count, bad_bacteria_count)
    for cell in environment["Inflammation"]:
        increase, cytokine_objects = Inflammation.cytokine_increase(cell, cytokine_objects)
        state['cytokine_level'] += increase*0.00001
    return state


def initialize_bacteria(microbe_trackers, num_good_bacteria, num_bad_bacteria):
    """
    Initialize the microbe_trackers with a starting number of good and bad bacteria.
    """
    # Initialize good bacteria
    for i in range(1, 5):  # Assuming you have 4 types of good bacteria
        species_name = f"Good_Bacteria_{i}"
        for _ in range(num_good_bacteria):
            location = (random.uniform(0, SIDE_LENGTH),
                        random.uniform(0, SIDE_LENGTH))
            microbe = Microbe(species=species_name,
                              location=location)
            microbe_trackers['good object'][species_name].append(microbe)

    # Initialize bad bacteria
    for i in range(1, 3):  # Assuming you have 2 types of bad bacteria
        species_name = f"Bad_Bacteria_{i}"
        for _ in range(num_bad_bacteria):
            location = (random.uniform(0, SIDE_LENGTH),
                        random.uniform(0, SIDE_LENGTH))
            microbe = Microbe(species=species_name,
                              location=location)
            microbe_trackers['bad object'][species_name].append(microbe)

# Function to handle microbe interactions


def replenish_bacteria(microbe_trackers, species):
    if "Good_Bacteria" in species:

        for i in range(50):
            location = (random.uniform(0, SIDE_LENGTH),
                        random.uniform(0, SIDE_LENGTH))
            microbe = Microbe(species, location)
            microbe_trackers['good object'][species].append(microbe)
    elif "Bad_Bacteria" in species:

        for i in range(50):
            location = (random.uniform(0, SIDE_LENGTH),
                        random.uniform(0, SIDE_LENGTH))
            microbe = Microbe(species, location)
            microbe_trackers['bad object'][species].append(microbe)


def combine_bacteria(microbe_trackers):
    """
    Combine all bacteria from different arrays and dictionaries into a single list.
    """
    combined_bacteria = []

    # Combine good bacteria from all species in "good object"
    for good_species, good_bacteria_list in microbe_trackers["good object"].items():
        combined_bacteria.extend(good_bacteria_list)

    # Combine bad bacteria from all species in "bad object"
    for bad_species, bad_bacteria_list in microbe_trackers["bad object"].items():
        combined_bacteria.extend(bad_bacteria_list)

    return combined_bacteria  # Return the combined list of all bacteria


def add_bacteria(new_bacterium, glycogen_objects, new_bacteria):
    # Only proceed if a new bacterium is successfully replicated
    if new_bacterium is not None and new_bacterium.species is not None:

        new_bacterium.interact_with_glycogen(glycogen_objects)
        new_bacterium.move()  # New bacterium moves after replication
        # Add new bacteria to the list
        new_bacteria.append(new_bacterium)


def handle_microbes(state, microbe_trackers, glycogen_objects, CA125_objects, bacteriocin_objects, bacteriocin_objects_E):
    """
    Process all bacteria by interacting with glycogen, moving, and attempting replication.
    Newly replicated bacteria are added to the appropriate species tracker.
    """

    if not any(microbe_trackers['good object'].values()) or not any(microbe_trackers['bad object'].values()):
        # Initialize with some starting bacteria
        initialize_bacteria(
            microbe_trackers, INITIAL_GOOD, INITIAL_BAD)

    #bacteriocins
    for bacteria in microbe_trackers["good object"]["Good_Bacteria_2"]:
        if random.randint(1,3) == 1:
            bacteriocin = Protein("Bacteriocin", bacteria.location)
            bacteriocin_objects.append(bacteriocin)
            
    for bacteria in microbe_trackers["good object"]["Good_Bacteria_3"]:
        if random.randint(1,3) == 1:
            bacteriocin = Protein("Bacteriocin", bacteria.location)
            bacteriocin_objects_E.append(bacteriocin)
    # Combine all existing bacteria into one list
    all_bacteria = combine_bacteria(microbe_trackers)

    # Shuffle the list for randomness in processing order
    random.shuffle(all_bacteria)
    try:

        good_bacteria_count = sum(microbe_trackers["good bacteria tracker"][species][-1]
                                  for species in microbe_trackers["good bacteria tracker"])
        bad_bacteria_count = sum(microbe_trackers["bad bacteria tracker"][species][-1]
                                 for species in microbe_trackers["bad bacteria tracker"])
        total_bacteria_count = good_bacteria_count + bad_bacteria_count
        good_proportion = good_bacteria_count/total_bacteria_count
    except:
        good_proportion = 0.5
    # Store new bacteria that will be added after processing
    new_bacteria = []
    replication_counter = 0
    # Process each bacterium
    move_glycogen(glycogen_objects)
    for bacterium in all_bacteria:
        if bacterium.alive:
            # Bacteria consume glycogen if nearby
            bacterium.interact_with_glycogen(glycogen_objects)
            bacterium.move()  # Bacteria move within the environment

            if replication_counter <= len(glycogen_objects):

                if "Good_Bacteria" in bacterium.species and good_proportion < random.uniform(0.5, 1):

                    if state['pH'] < 4.5 and state['pH'] > 3.5:
                        # Attempt replication based on nearby glycogen
                        new_bacterium = bacterium.replicate(glycogen_objects, CA125_objects)
                        add_bacteria(
                            new_bacterium, glycogen_objects, new_bacteria)
                        replication_counter += 1
                    else:
                        continue
                elif "Bad_Bacteria" in bacterium.species and good_proportion > random.uniform(0, 0.2):
                    # print(state["pH"])
                    if state['pH'] > 4:
                        replication_factor = random.randint(1, 2)
                        for i in range(replication_factor):
                            new_bacterium = bacterium.replicate(
                                glycogen_objects, CA125_objects)
                            add_bacteria(
                                new_bacterium, glycogen_objects, new_bacteria)
                            replication_counter += 1

                    else:
                        # inhibit or no growth
                        number_generated = random.randint(0, 1)
                        if number_generated == 0:
                            continue
                        else:
                            for i in range(number_generated):
                                new_bacterium = bacterium.replicate(
                                    glycogen_objects, CA125_objects)
                                add_bacteria(
                                    new_bacterium, glycogen_objects, new_bacteria)
                                replication_counter += 1

    # After all bacteria are processed, add the new bacteria to the trackers
    for new_bacterium in new_bacteria:
        if "Good_Bacteria" in new_bacterium.species:
            microbe_trackers['good object'][new_bacterium.species].append(
                new_bacterium)

        elif "Bad_Bacteria" in new_bacterium.species:
            microbe_trackers['bad object'][new_bacterium.species].append(
                new_bacterium)

    # Remove dead microbes from the trackers
    for good_species in microbe_trackers["good object"]:
        microbe_trackers["good object"][good_species] = [
            m for m in microbe_trackers["good object"][good_species] if m.alive]

    for bad_species in microbe_trackers["bad object"]:
        microbe_trackers["bad object"][bad_species] = [
            m for m in microbe_trackers["bad object"][bad_species] if m.alive]

    for good_species in microbe_trackers["good object"]:
        if not microbe_trackers["good object"][good_species]:
            # If the population of this good species is zero, replenish it
            print(f"Replenishing {good_species} population.")
            replenish_bacteria(microbe_trackers, good_species)

    for bad_species in microbe_trackers["bad object"]:
        if not microbe_trackers["bad object"][bad_species]:
            # If the population of this bad species is zero, replenish it
            print(f"Replenishing {bad_species} population.")
            replenish_bacteria(microbe_trackers, bad_species)
    for microbe in microbe_trackers["bad object"]["Bad_Bacteria_1"]:
        microbe.interact_with_bacteriocins(bacteriocin_objects, microbe_trackers["bad object"]["Bad_Bacteria_1"])
    for microbe in microbe_trackers["bad object"]["Bad_Bacteria_2"]:
        microbe.interact_with_bacteriocins(bacteriocin_objects_E, microbe_trackers["bad object"]["Bad_Bacteria_2"])


def generate_immune_cells(state, microbe_trackers, location):
    if location == 0:
        if state['estrogen_level'] > 0.6:
            no_new_immune_cells = int((NUMBER_IMMUNE_CELLS + 2) *
                                      (1+state['cytokine_level']))
            for _ in range(no_new_immune_cells):
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
                    
        elif state["progesterone_level"] > 0.6: 
            pass
        
        elif state["estrogen_level"] > 0.3:
            no_new_immune_cells = int(NUMBER_IMMUNE_CELLS *
                                      (1+state['cytokine_level']))
            for _ in range(no_new_immune_cells):
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

                microbe_trackers['immune cells'].append(new_immune_cell)
    else:
        new_immune_cell = Neutrophil(location)
        microbe_trackers['immune cells'].append(new_immune_cell)
        
    microbe_trackers['immune production'].append(len(microbe_trackers['immune cells']))

        
# Function to handle immune cell interactions


def calculate_distance(location1, location2):
    return np.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)


def handle_immune_cells(microbe_trackers):

    all_bacteria = combine_bacteria(microbe_trackers)
    microbe_trackers['good object'] = {
        f"Good_Bacteria_{i}": [] for i in range(1, 5)}
    microbe_trackers['bad object'] = {
        f"Bad_Bacteria_{i}": [] for i in range(1, 3)}

    # Shuffle the list for randomness in processing order
    random.shuffle(all_bacteria)

    for immune_cell in microbe_trackers['immune cells'][:]:
        killed = False

        if isinstance(immune_cell, Neutrophil):
            killed, bacteria_list = immune_cell.interact(all_bacteria)

        elif isinstance(immune_cell, Macrophage):
            killed, bacteria_list = immune_cell.interact(all_bacteria)

        elif isinstance(immune_cell, NKcell):
            killed, all_bacteria = immune_cell.interact(all_bacteria)

        if killed or immune_cell.age_cell() or immune_cell.move(all_bacteria):
            microbe_trackers['immune cells'].remove(immune_cell)

    for bacterium in all_bacteria:
        if "Good_Bacteria" in bacterium.species:
            microbe_trackers['good object'][bacterium.species].append(
                bacterium)
        if "Bad_Bacteria" in bacterium.species:
            microbe_trackers['bad object'][bacterium.species].append(bacterium)

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


def create_heatmap(dictionary, entity_type="bacteria together"):
    # Initialize an empty grid with the correct number of bins
    heatmap = np.zeros((num_bins, num_bins))

    if entity_type == "bacteria together":
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

    elif entity_type == "good bacteria":
        for good_species in dictionary["good object"]:
            for bacterium in dictionary["good object"][good_species]:
                x_bin, y_bin = get_bin(bacterium.location)
                # Increment count in the correct bin
                heatmap[y_bin, x_bin] += 1

    elif entity_type == "bad bacteria":
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
    
    elif entity_type == "Inflammation":
        for cell in dictionary[entity_type]:
            x_bin, y_bin = get_bin(cell.location)
            heatmap[y_bin, x_bin] = cell.measure
    
    else:
        for cell in dictionary[entity_type][-1]:
            x_bin, y_bin = get_bin(cell.location)
            heatmap[y_bin, x_bin] += 1
        
    return heatmap


def plot_heatmap(microbe_trackers, environment):

    # First figure: "Bacteria Distribution" and "Glycogen Distribution"
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 6))

    # Create each heatmap for the first figure
    bacteria_heatmap = create_heatmap(
        microbe_trackers, entity_type="bacteria together")
    glycogen_heatmap = create_heatmap(environment, entity_type="glycogen")

    # Plot "Bacteria Distribution" heatmap in the first subplot
    im1 = axs1[0].imshow(bacteria_heatmap, cmap="Reds", interpolation='nearest',
                         origin='lower', extent=[0, grid_size, 0, grid_size])
    axs1[0].set_title("Bacteria Distribution Heatmap")
    fig1.colorbar(im1, ax=axs1[0], label='Count')

    # Plot "Glycogen Distribution" heatmap in the second subplot
    im2 = axs1[1].imshow(glycogen_heatmap, cmap="Greens", interpolation='nearest',
                         origin='lower', extent=[0, grid_size, 0, grid_size])
    axs1[1].set_title("Glycogen Distribution Heatmap")
    fig1.colorbar(im2, ax=axs1[1], label='Count')

    # Set labels for each axis in the first figure
    for ax in axs1:
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

    # Adjust layout for the first figure
    plt.tight_layout()
    plt.show()

    # Second figure: "Good Bacteria Distribution" and "Bad Bacteria Distribution"
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))

    # Create each heatmap for the second figure
    good_bacteria_heatmap = create_heatmap(
        microbe_trackers, entity_type="good bacteria")
    bad_bacteria_heatmap = create_heatmap(
        microbe_trackers, entity_type="bad bacteria")

    # Plot "Good Bacteria Distribution" heatmap in the first subplot
    im3 = axs2[0].imshow(good_bacteria_heatmap, cmap="Purples",
                         interpolation='nearest', origin='lower', extent=[0, grid_size, 0, grid_size])
    axs2[0].set_title("Good Bacteria Distribution")
    fig2.colorbar(im3, ax=axs2[0], label='Count')

    # Plot "Bad Bacteria Distribution" heatmap in the second subplot
    im4 = axs2[1].imshow(bad_bacteria_heatmap, cmap="Oranges", interpolation='nearest',
                         origin='lower', extent=[0, grid_size, 0, grid_size])
    axs2[1].set_title("Bad Bacteria Distribution")
    fig2.colorbar(im4, ax=axs2[1], label='Count')

    # Set labels for each axis in the second figure
    for ax in axs2:
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")


def plot_heatmap_seperate(heatmap, title="Bacteria Distribution", cmap="hot"):

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap=cmap, interpolation='nearest',
               origin='lower', extent=[0, grid_size, 0, grid_size])
    plt.colorbar(label='Count')

    #plt.legend()
    plt.title(title)
    plt.xlabel('X Position ')
    plt.ylabel('Y Position')
    plt.show()


def plotting(environment, microbe_trackers, good_bacteria_proportion_tracker, tumour_tracker):
    '''# Plotting the results
    plt.figure(figsize=(12, 10))

    plt.subplot(7, 1, 1)
    plt.plot(range(1, len(environment["estrogen"]) + 1),
             environment["estrogen"], label='Estrogen Levels', linestyle='--')
    plt.plot(range(1, len(environment["progesterone"]) + 1),
             environment["progesterone"], label='Progesterone Levels', linestyle='--')
    plt.title('Hormone Levels Over Time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Hormone Levels')
    plt.legend()

    plt.subplot(7, 1, 2)

    plt.plot(range(1, len(environment["glycogen"]) + 1), environment["glycogen"],
             label='Gylcogen Level', linestyle='--', color='blue')

    plt.title('Total Glycogen Levels over time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Level')
    plt.legend()

    plt.subplot(7, 1, 3)
    plt.plot(range(1, len(environment["iron"]) + 1), environment["iron"],
             label='Iron Levels', linestyle='--', color='orange')
    plt.plot(range(1, len(environment["h2o2"]) + 1), environment["h2o2"],
             label='H2O2 Levels', linestyle='--', color='blue')
    plt.plot(range(1, len(environment["cytokine"]) + 1), environment["cytokine"],
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
    plt.legend(

    plt.subplot(7, 1, 5)
    for good_species in microbe_trackers['good bacteria tracker']:
        plt.plot(range(1, len(microbe_trackers['good bacteria tracker'][good_species]) + 1),
                 microbe_trackers['good bacteria tracker'][good_species], label=f'{good_species} Count')
    plt.title('Good Bacteria Produced Totals')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(7, 1, 6)
    for bad_species in microbe_trackers['bad bacteria tracker']:
        plt.plot(range(1, len(microbe_trackers['bad bacteria tracker'][bad_species]) + 1),
                 microbe_trackers['bad bacteria tracker'][bad_species], label=f'{bad_species} Count')
    plt.title('Bad Bacteria Produced Totals')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Count')
    plt.legend()


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
             microbe_trackers["immune tracker"], label="immune cells at end of step")
    plt.plot(range(1, len(microbe_trackers["immune production"]) + 1),
             microbe_trackers["immune production"], label="immune at start of step")
    plt.title('Immune cells')
    plt.xlabel('Simulation Steps')
    plt.ylabel('# of immune cells')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fname="plots", dpi=1000)
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.hist(good_bacteria_proportion_tracker, bins=5,
             density=False, label="Distribution of Trials")
    plt.title('Histogram of Good Bacteria for Each Trial')
    plt.xlabel('Proportion of Good Bacteria')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(fname="Distribution of trials", dpi=1000)
    plt.show()
'''

    # Plotting for Report

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(environment["estrogen"]) + 1),
             environment["estrogen"], label='Estrogen Levels', linestyle='--')
    plt.plot(range(1, len(environment["progesterone"]) + 1),
             environment["progesterone"], label='Progesterone Levels', linestyle='--')
    plt.plot(range(1, len(microbe_trackers['proportion'])+1),
             microbe_trackers['proportion'], label='Proportion of good bacteria',
             linestyle='--')
    plt.title('Levels Over Time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Levels')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(environment["iron"]) + 1), environment["iron"],
             label='Iron Levels', linestyle='--', color='orange')

    plt.title('Iron Pulses')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Iron Levels')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fname="Hormone and Iron Levels.png", dpi=1000)
    plt.show()

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(environment["glycogen"]) + 1), environment["glycogen"],
             label='Glycogen Level', linestyle='--', color='blue')

    plt.title('Total Glycogen Levels over time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Number of clumps')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(environment["pH"]) + 1), environment["pH"],
             label='pH Levels', linestyle='--', color='blue')
    plt.plot(range(1, len(environment["cytokine"]) + 1), environment["cytokine"],
             label='Cytokine level', linestyle='--', color='green')
    plt.legend()


    plt.subplot(3, 1, 3)
    plt.plot(range(1, len(microbe_trackers["total bad"]) + 1),
             microbe_trackers["total bad"], label='Bad Bacteria', linestyle='--')
    plt.plot(range(1, len(microbe_trackers["total good"]) + 1),
             microbe_trackers["total good"], label='Good Bacteria', linestyle='--')
    plt.plot(range(1, len(microbe_trackers["total bacteria"]) + 1),
             microbe_trackers["total bacteria"], label='Total Bacteria', linestyle='--')
    plt.title('Total Microbes')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Number of Bacteria')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fname="Glycogen and Bacteria Quantity", dpi=100)
    plt.show()
    
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(environment["CA125_level"]) + 1), environment["CA125_level"],
             label='CA125 Levels', linestyle='--', color='blue')
    plt.plot(range(1, len(environment["VEGF_level"]) + 1), environment["VEGF_level"],
             label='VEGF levels', color='red')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 10))
    plt.hist(good_bacteria_proportion_tracker, bins=10,
             density=False, label="Distribution of Trials")
    plt.title('Histogram of Good Bacteria for Each Trial')
    plt.xlabel('Proportion of Good Bacteria')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(fname="Distribution of trials", dpi=1000)
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(microbe_trackers["immune tracker"]) + 1),
             microbe_trackers["immune tracker"], label="immune cells at end of step")
    plt.plot(range(1, len(microbe_trackers["immune production"]) + 1),
             microbe_trackers["immune production"], label="immune at start of step")
    plt.title('Immune response')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(fname="immune_responce.png", dpi=1000)
    plt.show()
    
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, len(tumour_tracker)+1), tumour_tracker, label="Tumour development")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10,6))
    for good_species in microbe_trackers['good bacteria tracker']:
        microbe_trackers['good bacteria tracker'][good_species].remove(microbe_trackers['good bacteria tracker'][good_species][0])
        plt.plot(range(1, len(microbe_trackers['good bacteria tracker'][good_species]) + 1),
                 microbe_trackers['good bacteria tracker'][good_species], label=f'{good_species} Count')
    plt.title('Good Bacteria Produced Totals')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Count')
    plt.legend()

    for bad_species in microbe_trackers['bad bacteria tracker']:
        microbe_trackers['bad bacteria tracker'][bad_species].remove(microbe_trackers['bad bacteria tracker'][bad_species][0])
        plt.plot(range(1, len(microbe_trackers['bad bacteria tracker'][bad_species]) + 1),
                 microbe_trackers['bad bacteria tracker'][bad_species], label=f'{bad_species} Count')
    plt.legend()
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
            "pH": [],
            "cytokine": [],
            "CA125_objects": [],
            "CA125_level": [],
            "Inflammation": [Inflammation([i,j], 0)
                                    for i in range(0,10) for j in range(0,10)],
            "VEGF_level": [],
            "Bacteriocins": [],
            "Bacteriocins_E": []
        }

        # Initialize bacteria trackers
        microbe_trackers = {
            "good bacteria tracker": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
            "bad bacteria tracker": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)},
            "good bacteria step": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
            "bad bacteria step": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)},
            "immune cells": [],
            "immune tracker": [],
            "immune production": [],
            "good bacteria step plot": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
            "bad bacteria step plot": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)},
            "good object": {f"Good_Bacteria_{i}": [] for i in range(1, 5)},
            "bad object": {f"Bad_Bacteria_{i}": [] for i in range(1, 3)}
        }

        total_good_bacteria = []
        total_bad_bacteria = []
        total_bacteria = []
        tumour_tracker = []
        
        proportion_tracker_over_time = []

        # Initialize persistent microbes and objects
        good_bacteria_count = 0
        bad_bacteria_count = 0
        glycogen_objects = []  # Glycogen objects initialized
        CA125_objects = []
        inflammation_objects = environment["Inflammation"]
        cytokine_objects = []
    

        tumour = generate_tumour()
        for cycle in range(NUM_CYCLES):
            print(f"Cycle {cycle}")
            for step in range(DAYS):

                # Get hormone levels for the current step
                estrogen_level = estrogen_levels[step]
                progesterone_level = progesterone_levels[step]
                pH_levels = environment["pH"]
                bacteriocin_objects = []
                bacteriocin_objects_E = []

                # Initialize the state for the current step, passing existing glycogen_objects
                current_state = initialize_state(
                    estrogen_level, progesterone_level, good_bacteria_count, 
                    bad_bacteria_count, glycogen_objects, step, pH_levels, 
                    tumour, CA125_objects, inflammation_objects, cycle)

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
                # Store pH level
                environment["pH"].append(
                    current_state['pH'])
                environment["CA125_level"].append(current_state["CA125_level"])
                environment["CA125_objects"].append([CA for CA in current_state["CA125_objects"]])
                environment["Inflammation"] = current_state["Inflammation"]
                environment["VEGF_level"].append(current_state["VEGF_level"])
                
                
                # Handle microbe interactions (including glycogen consumption)
                handle_microbes(
                    current_state, microbe_trackers, glycogen_objects, CA125_objects, bacteriocin_objects, bacteriocin_objects_E)
                environment["Bacteriocins"].append(bacteriocin_objects)
                environment["Bacteriocins_E"].append(bacteriocin_objects_E)
                # Generate new immune cells
                generate_immune_cells(current_state, microbe_trackers, 0)
                Inflammatory_response(microbe_trackers, cytokine_objects)

                # Handle immune cell interactions
                handle_immune_cells(microbe_trackers)
                # Update step trackers
                update_trackers(microbe_trackers)

                # Update chemical levels in the state
                update_chemical_levels(environment,
                                       current_state, good_bacteria_count, bad_bacteria_count, step, cytokine_objects)

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

                current_good_proportion = good_bacteria_count/total_bacteria_count
                proportion_tracker_over_time.append(current_good_proportion)
                tumour.growth(current_state["VEGF_level"])
                tumour_tracker.append(tumour.size)#np.append(tumour_tracker,[tumour.age, tumour.size])
               # print(tumour_tracker)

        # Track good bacteria proportions at the end of the trial
        good_bacteria_proportion = good_bacteria_count / \
            (good_bacteria_count + bad_bacteria_count)
        good_bacteria_proportion_tracker.append(good_bacteria_proportion)

        # Add total bacteria to the microbe_trackers
        microbe_trackers["total good"] = total_good_bacteria
        microbe_trackers["total bad"] = total_bad_bacteria
        microbe_trackers["total bacteria"] = total_bacteria
        microbe_trackers['proportion'] = proportion_tracker_over_time

    return environment, microbe_trackers, good_bacteria_proportion_tracker, tumour_tracker, tumour


def main():
    environment, microbe_trackers, good_bacteria_proportion_tracker, tumour_tracker, tumour = simulation_loop()
    plotting(environment, microbe_trackers, good_bacteria_proportion_tracker, tumour_tracker)

    # Calculate and print the average proportion of good bacteria across all trials
    print(
        f"Average Proportion of good bacteria over {TRIALS} TRIALS is: {np.average(good_bacteria_proportion_tracker):.2f}")

    # Calculate and print the standard deviation of the proportion of good bacteria across
    # Calculate and print the standard deviation of the proportion of good bacteria across all trials
    standard_deviation_proportion = np.std(good_bacteria_proportion_tracker)
    print(
        f"The standard deviation of the trials is: {standard_deviation_proportion:.2f}")

    plot_heatmap(microbe_trackers, environment)
    immune_cell_heatmap = create_heatmap(
        microbe_trackers, entity_type="immune_cells")
    plot_heatmap_seperate(immune_cell_heatmap,
                          title="Immune Cell Distribution Heatmap", cmap="Blues")
    bacteriocin_heatmap = create_heatmap(
        environment, entity_type="Bacteriocins")
    plot_heatmap_seperate(bacteriocin_heatmap,
                          title="Bacteriocin Distribution Heatmap", cmap="Blues")
    CA125_heatmap = create_heatmap(
        environment, entity_type="CA125_objects")
    plot_heatmap_seperate(CA125_heatmap,
                          title="CA125 Distribution Heatmap", cmap="Blues")
    Inflammation_heatmap = create_heatmap(
        environment, entity_type="Inflammation")
    plot_heatmap_seperate(Inflammation_heatmap,
                          title="Inflammation Distribution Heatmap", cmap="Blues")
    print("Tumour Grade:" + str(tumour.grade), "Tumour Stage:" + str(tumour.stage))
    print("Asymptomatic:"+str(tumour.asymptomatic), "Growth Speed:"+ tumour.speed)
    print(f"Initial State: Stage: {tumour.initial[0]:1g}, Grade:{tumour.initial[1]:1g}, Size: {tumour.initial[2]:1g}, Age: {tumour.initial[3]:2g}")
    
    return environment, microbe_trackers, good_bacteria_proportion_tracker


environment, microbe_trackers, good_bacteria_proportion_tracker = main()