import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


nnbf_data = pd.read_csv("TrainAnalysis/energy_log_adam_sr_7up_7dn_1.5relerror.csv")
fnbf_data = pd.read_csv("TrainAnalysis/training_log_fnbf_look_good.csv")

plt.plot(nnbf_data["iter"][:5000], nnbf_data["energy"][:5000], label="NNBF")
plt.plot(fnbf_data["step"], fnbf_data["energy"], label="FNBF")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Energy vs Iteration")
plt.legend()
plt.savefig("TrainAnalysis/energy_vs_iteration.png")
plt.show()