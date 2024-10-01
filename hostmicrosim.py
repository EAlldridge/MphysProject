# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:47:51 2024

@author: Beth
"""

import random
import matplotlib.pyplot as plt

class Microbe:
    def __init__(self, species, location):
        self.species = species
        self.location = location

    def move(self):
        new_location = (self.location[0] + random.uniform(-1, 1), self.location[1] + random.uniform(-1, 1))
        self.location = new_location

    def interact(self, estrogen_level, progesterone_level):
        # Simulate microbiome response to hormone levels
        hormone_level = (estrogen_level + progesterone_level) / 2  # Average of estrogen and progesterone
        if hormone_level > 0.5:  # Higher hormone levels favor good bacteria
            self.species = "Good Bacteria"
        else:
            self.species = "Bad Bacteria"

num_microbes = 5
simulation_steps = 10

estrogen_levels = []
progesterone_levels = []
good_bacteria_abundance = []
bad_bacteria_abundance = []

for step in range(simulation_steps):
    print(f"Step {step + 1}:")

    # Simulate hormonal changes during the menstrual cycle
    estrogen_level = random.uniform(0, 1)
    progesterone_level = random.uniform(0, 1)
    estrogen_levels.append(estrogen_level)
    progesterone_levels.append(progesterone_level)
    print(f"Estrogen Level: {estrogen_level}")
    print(f"Progesterone Level: {progesterone_level}")

    # Microbe interactions and abundance based on hormone levels
    good_bacteria_count = 0
    bad_bacteria_count = 0
    for i in range(num_microbes):
        microbe = Microbe(species=f"Microbe_{i}", location=(random.uniform(0, 10), random.uniform(0, 10)))
        microbe.move()
        microbe.interact(estrogen_level, progesterone_level)
        if microbe.species == "Good Bacteria":
            good_bacteria_count += 1
        elif microbe.species == "Bad Bacteria":
            bad_bacteria_count += 1

    good_bacteria_abundance.append(good_bacteria_count)
    bad_bacteria_abundance.append(bad_bacteria_count)
    print(f"Good Bacteria Abundance: {good_bacteria_count}")
    print(f"Bad Bacteria Abundance: {bad_bacteria_count}")

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(range(1, simulation_steps + 1), estrogen_levels, label='Estrogen Levels', linestyle='--')
plt.plot(range(1, simulation_steps + 1), progesterone_levels, label='Progesterone Levels', linestyle='--')
plt.title('Hormone Levels Over Time')
plt.xlabel('Simulation Steps')
plt.ylabel('Hormone Levels')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(1, simulation_steps + 1), good_bacteria_abundance, label='Good Bacteria Abundance')
plt.plot(range(1, simulation_steps + 1), bad_bacteria_abundance, label='Bad Bacteria Abundance')
plt.title('Microbiome Dynamics Over Time')
plt.xlabel('Simulation Steps')
plt.ylabel('Abundance')
plt.legend()

plt.tight_layout()
plt.show()