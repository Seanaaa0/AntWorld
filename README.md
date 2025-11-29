# AntWorldSim — Multi-Stage Ant Colony Simulation
<img width="877" height="961" alt="image" src="https://github.com/user-attachments/assets/5258192b-f016-4329-b9fb-0fb51aeb758a" />

This repository contains three evolutionary versions of the AntWorld simulation.
Each stage upgrades ant intelligence, memory, communication, and return-to-food behavior.
All versions share the same visualization system (main_visual_stage3.py).

---

## Versions Overview

### Stage 3 — Base Intelligence
Files:
- envs/env_interface_3.py
- antagent/AntAgent2.py
- main_visual_stage3.py

Features:
- Stable movement
- Avoids nest deadlocks
- Exploration and memory
- Returns home after max steps
- First stable version

---

### Stage 4 — Reproduction Version
Files:
- envs/env_interface_4.py
- antagent/AntAgent3.py
- main_visual_stage3.py

Features:
- Bringing food back creates one new ant
- Nest size expands based on population
- Food scent (pheromone-like) added
- Improved foraging efficiency

---

### Stage 5 — Communication and Advanced Memory
Files:
- envs/env_interface_5.py
- antagent/AntAgent4.py
- main_visual_stage3.py

Features:
- Ants communicate within a 5x5 radius
- After delivering food, ants return to the original food source
- Strong spatial memory and scent following
- Collective intelligence behavior emerges
- Faster food cluster discovery

---

## Directory Structure

AntWorld/
  antagent/
    AntAgent2.py
    AntAgent3.py
    AntAgent4.py
  envs/
    env_interface_3.py
    env_interface_4.py
    env_interface_5.py
  main_visual_stage3.py
  README.md

---

## Running

python main_visual_stage3.py

To switch versions, change inside the script:

from env_interface_5 import AntSimInterface

Options:

from env_interface_3 import AntSimInterface
from env_interface_4 import AntSimInterface
from env_interface_5 import AntSimInterface

---

## Requirements

numpy
pygame

Install:

pip install numpy pygame

---

## Project Summary

AntWorldSim is a multi-agent ant colony environment designed for simulation, agent behavior studies, and future reinforcement learning experiments. Across stages, the ants evolve in memory, communication, and adaptive behavior. The simulation separates environment logic, agent logic, and visualization to maintain clarity and modularity.
