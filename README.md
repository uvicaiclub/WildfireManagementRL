# WildfireManagementRL 🔥
Simulating Responsive Action to Wildfire Disasters with Reinforcement Learning (RL)

---
## 💖Motivation:<br> 
Forest Fires are a major source of air pollution, release vast amounts of CO2, and risk damaging human infrastructure. The costs of Forest Fire fighting in BC is closing in on $ 1 billion/ year and increasing yearly [source](https://www2.gov.bc.ca/gov/content/safety/wildfire-status/about-bcws/wildfire-history/wildfire-season-summary).

---
## 📝Background:<br> 
There exists a wide range of fire spread simulators. These take into consideration elevation, wind, fuel types, and are important for developing a management plan. In these simulations, however, human intervention is not being considered in the spread dynamics.

Previous work has attempted to introduce RL agents to fight fires, but their simulations and agent actions do not sufficiently reflect reality.

---
## 💡Our contributions:<br> 
- We are developing an environment informed by modern fire spread simulators.
- Parameters involved in spreading dynamics will be tuned to mirror realistic fire spread for the Northwestern BC region.
- Multiple RL agents are trained to fight a fire together. The actions the agent can take are informed by fire crew operations.


Wildfire Simulation with AI Agents

This project investigates the use of artificial intelligence (AI) agents in a wildfire simulation environment. The goal is to develop a model that can be used to evaluate different firefighting tactics and strategies.

The simulation incorporates several factors that influence fire spread, including:

    Fuel moisture
    Wind direction and intensity
    Fire intensity

AI agents are introduced into the simulation to model firefighting efforts. These agents use a decision-making process to target areas for water application based on the environmental data mentioned above.
Key Findings

    Heuristic agents that target fire spread based on wind and dryness can significantly improve fire containment compared to a scenario without any agents.
    Spotting behavior, where embers jump fire lines and ignite new areas, is a challenge to model realistically in the simulation.

Future Work

    The project aims to increase the fidelity of the simulation by incorporating additional factors such as:
        Different fuel types
        Variable wind patterns
        Elevation data
    Agent realism will be improved through:
        Higher temporal resolution
        More diverse interaction methods
        Reinforcement learning for more nuanced decision-making

This project is a work in progress, and the findings outlined here are based on the current state of development.
