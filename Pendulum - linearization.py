import control as ct
import numpy as np
import math
import matplotlib.pyplot as plt
import pycollimator as collimator

s = ct.TransferFunction.s

# System parameters
m = 0.5
l = 1
k = 0.5
J = m*l*l
g = 9.81

# Transfer function (linearised)
P = 1/(J*s**2 + k*s - m*g*l)

# Controller transfer function
tau = 0.2
z = -3
K = 2
C = K*(s-z)/(tau*s+1)

# Convert to PID gains
Kp = -z*K
Ki = 0
Kd = K+z*K*tau

# Close-loop transfer function
G = ct.feedback(C*P, 1)

# Plot root-locus of P
plt.figure()
ct.root_locus(P)

# Plot root-locus of C*P
plt.figure()
ct.root_locus(C*P)

# Plot Bode diagram
plt.figure()
ct.nyquist_plot(C*P)

# Ideal step response
t, yout = ct.step_response(G)
plt.figure()
plt.plot(t, yout)
plt.grid()

# Load token for Collimator from file
token_file = open("token.txt", 'r')
token = token_file.read()

# Load model from Collimator
project_uuid = "221181d1-4494-4cf8-a58a-61345a7aa14c"
collimator.set_auth_token(token, project_uuid)
model = collimator.load_model("Pendulum - Linearized PID Control")

# Create array of mass values for robustness analysis
m_V = [0.4, 0.5, 0.6]

# Create array of results
res_V = []

# Simulate for each mass value and store results in res_V
for m in m_V:
    sim = collimator.run_simulation(model, parameters = {'m': m, 'Kp':Kp, 'Ki':Ki, 'Kd': Kd, 'T_c':tau, 'Theta_0': 175})
    res = sim.results.to_pandas()
    res_V.append(res)

# Create figure
plt.figure()

# Plot response for each mass value and setpoint in 1st subplot
plt.subplot(2, 1, 1)

for idx in range(len(m_V)):
    plt.plot(res_V[idx].index, res_V[idx]["Pendulum.Theta"], label=f"Response - Mass = {m_V[idx]} kg")

plt.plot(res_V[0].index, res_V[0]["Setpoint.out_0"], "--", label="Setpoint")
plt.xlabel("Time [s]")
plt.ylabel("Theta [deg]")
plt.legend()
plt.grid()

# Plot controller output for each mass value
plt.subplot(2, 1, 2)

for idx in range(len(m_V)):
    plt.plot(res_V[idx].index, res_V[idx]["Discrete_PID.Command"], label=f"Tau - Mass = {m_V[idx]} kg")

plt.xlabel("Time [s]")
plt.ylabel("Tau [Nm]")
plt.legend()
plt.grid()

# Show plots
plt.show()