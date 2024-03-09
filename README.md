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
- Adding heuristic-based agents will more accurately model the push-and-pull dynamic of real-world wildfire spread in BC.
- Multiple RL agents are trained to fight a fire together. The actions the agent can take are informed by fire crew operations.
