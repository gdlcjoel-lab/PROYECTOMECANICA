"""
bonus_analysis.py — Entry point for the 3-stage mission bonus exercise.

Run:
    python bonus_analysis.py

Produces:
    - Mission summary printed to console
    - Figure 1: Timeline (altitude, speed, mass, phase portrait)
    - Figure 2: Orbital view (ECI x-y plane)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from bonus_multistage import simulate_multistage, print_mission_summary
from bonus_plot import plot_mission, plot_orbit


def main():
    print('Running 3-stage simulation...')
    data = simulate_multistage(dt_atm=0.1, dt_space=5.0)

    print_mission_summary(data)

    fig1 = plot_mission(data)
    fig2 = plot_orbit(data)

    print('\nClose plot windows to exit.')
    plt.show()


if __name__ == '__main__':
    main()
