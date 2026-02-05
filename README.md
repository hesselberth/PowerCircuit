# PowerCircuit

Lcapy extension for simulating dcdc converters that combine linear circuit components (L, C, R, transformer) with sources (I, V) and non-linear components (switches and diodes). Switches have a fixed Rds_on and diodes are simplified to a piecewise linear element consisting of a forward voltage drop Vf and a diode resistance Rd. The simulator is a backward-Euler ODE integrator.
