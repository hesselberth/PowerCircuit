#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:05:58 2026

@author: hessel

Library to simulate power electronic circuits comprising of sources, linear
circuit elements like resistors, capacitors and inductors and nonlinear
elements like diodes and switches. The library takes the MNA (modified nodal
analysis) from LCapy and expands it to a switching circuit.
"""


import numpy as np
from lcapy import Circuit, pprint


# Default switching values, SI units
Rds_on = 0.01  # SWitch on
Rds_off = 1e7  # SWitch off is modeled as a high resistance state
Vf = 0.7       # Diode voltage drop
Rdf = 0.05     # Diode forward resistance
Rdb = 1e6      # Diode blocking is modeled as a resistance state


def bool_pow(n, prev=[]):
    """
    Boolean powerset of size n, where n is an integer.
    Creates a set of lists of boolean variables with all distinct possibilities.
    Returns a list containing 2 ** n boolean lists of length n.
    Used to manage switch states (on or off) and diode states (conducting or
    blocking).
    """
    if n <= 0:
        return [prev]
    else:
        t = prev.copy()
        t.append(True)
        f = prev.copy()
        f.append(False)
        return bool_pow(n-1, f) + bool_pow(n-1, t)


"""
Class implementing a nonlinear circuit containing switches and diodes.
Because of the nonlinear nature of these circuits algebraic analysis is
generally impossible. To increase computational performance, all sympy
matrices are converted to numpy. This means that no component values may be
left unspecified.
"""
class PowerCircuit:
    def __init__(self, netlist, **kwargs):
        self.netlist = netlist

        self.Rds_on = 0.01
        self.Rds_off = 1e7
        self.Vf = 0.7
        self.Rdf = 0.05
        self.Rdb = 1e6

        self.expanded_netlist, self.symbolic_netlist, self.switch_db, \
            self.diode_db = self.expand_netlist(netlist)
        self.expanded_circuit = Circuit(self.expanded_netlist)
        self.symbolic_circuit = Circuit(self.symbolic_netlist)
        self.expanded_statespace = self.expanded_circuit.state_space()
        self.symbolic_statespace = self.symbolic_circuit.state_space()
        assert(len(self.expanded_statespace.u) == len(self.symbolic_statespace.u))
        assert(len(self.expanded_statespace.x) == len(self.symbolic_statespace.x))
        assert(len(self.expanded_statespace.y) == len(self.symbolic_statespace.y))
        self.switch_list = self.switch_db.keys()
        self.diode_list = self.diode_db.keys()
        self.switch_R = [self.switch_db[switch] for switch in self.switch_list]
        self.diode_V = [self.diode_db[diode][0] for diode in self.diode_list]
        self.diode_R = [self.diode_db[diode][1] for diode in self.diode_list]
        self.diode_int = [self.diode_db[diode][2] for diode in self.diode_list]
        self.expanded_inputs = [str(inp) for inp in self.symbolic_statespace.u]
        #self.diode_V_input_indices = [self.expanded_inputs.index(Vstr) for Vstr in self.diode_V]
        self.y = [inp for inp in self.expanded_inputs if str(inp) not in self.diode_V]
        self.y = [inp for inp in self.expanded_inputs]
        self.y = np.array([self.y]).T
        print("y:")
        print(self.y)
        print("expanded_statespace.u:")
        print(self.expanded_statespace.u)
        print("diode_V_input_indices:")
        #print(self.diode_V_input_indices)
        self.input_indices = [self.expanded_inputs.index(inp) for inp in self.y]
        print("input_indices:")
        print(self.input_indices)
        self.expanded_outputs = [str(outp)[:-3] for outp in self.symbolic_statespace.y]
        print("expanded_outputs:")
        print(self.expanded_outputs)
        self.diode_I_V_output_indices = [self.expanded_outputs.index("i_"+outp) for outp in self.diode_V]
        print("diode_I_V_output_indices:")
        print(self.diode_I_V_output_indices)
        self.diode_I_R_output_indices = [self.expanded_outputs.index("i_"+outp) for outp in self.diode_R]
        print("diode_I_R_output_indices:")
        print(self.diode_I_R_output_indices)
        self.diode_v_int_output_indices = [self.expanded_outputs.index("v_"+outp) for outp in self.diode_int]
        print("diode_v_int_output_indices:")
        print(self.diode_v_int_output_indices)
        self.num_switches = len(self.switch_list)
        self.num_diodes = len(self.diode_list)
        self.switch_addr = 2 ** np.arange(self.num_switches, dtype = int)[::-1]
        self.diode_addr = 2 ** np.arange(self.num_diodes, dtype = int)[::-1]
        self.create_matrix()


    def expand_netlist(self, netlist_str):
        """
        Replaces 'Dname n1 n2' with a series R and V source and
        SWname n1 n2 by R.
        """
        lines = netlist_str.strip().split('\n')
        new_netlist = []
        new_symbolic_netlist = []
        switches = {}
        diodes = {}
        
        for line in lines:
            line = line.strip()
            parts = line.split()
            if not parts: continue
            
            # Check if the component is a diode (starts with D)
            if parts[0].startswith('D') and len(parts) >= 3:
                name = parts[0]           # e.g., D1
                anode = parts[1]          # e.g., 2
                cathode = parts[2]        # e.g., 3
                int_node = f"int_{name}"  # Internal node
                
                # Create the V_f source and the series resistance
                new_netlist.append(f"R_{name} {anode} {int_node} {{R_{name}}}")
                new_netlist.append(f"V_{name} {int_node} {cathode} {self.Vf}")
                new_symbolic_netlist.append(f"R_{name} {anode} {int_node}")
                new_symbolic_netlist.append(f"V_{name} {int_node} {cathode}")
                diodes[name] = (f"V_{name}", f"R_{name}", f"int_{name}")
            elif parts[0].startswith('SW') and len(parts) >= 3:
                name = parts[0]           # e.g., SW1
                Np = parts[1]             # e.g., 2
                Nm = parts[2]             # e.g., 3
                
                # Create the switch resistance
                new_netlist.append(f"R_{name} {Np} {Nm} {{R_{name}}}")
                new_symbolic_netlist.append(f"R_{name} {Np} {Nm} {{R_{name}}}")
                switches[name] = (f"R_{name}")
            else:
                new_netlist.append(line)
                new_symbolic_netlist.append(f"{parts[0]} {parts[1]} {parts[2]}")
        return '\n'.join(new_netlist), '\n'.join(new_symbolic_netlist), switches, diodes


    def enum_switch_states(self):
        Rsw_on = self.Rds_on
        Rsw_off = self.Rds_off
        Rd_on = self.Rdf
        Rd_off = self.Rdb

        switch_state_list = bool_pow(self.num_switches) if self.num_switches > 0 else []
        #print("sw", switch_state_list)
        switch_state_array = np.array(switch_state_list, dtype=int)
        #print("sw", switch_state_array)
        i_switch_state = np.dot(switch_state_array, self.switch_addr)
        #print("swi", i_switch_state)
        diode_state_list = bool_pow(self.num_diodes) if self.num_diodes > 0 else []
        #print("d", diode_state_list)
        diode_state_array = np.array(diode_state_list, dtype=int)
        #print("d", diode_state_array)
        i_diode_state = np.dot(diode_state_array, self.diode_addr)
        #print("di", i_diode_state)
        statedb = []
        if isinstance(i_switch_state, np.ndarray):
            for i , switch_addr in enumerate(i_switch_state):
                switch_addr = int(switch_addr)  # get rid of np.int64
                switch_state = switch_state_array[i]
                Rsw = Rds_on * switch_state + Rds_off * (1 - switch_state)
                if isinstance(i_diode_state, np.ndarray):
                    for j , diode_addr in enumerate(i_diode_state):
                        diode_addr = int(diode_addr)  # get rid of np.int64
                        diode_state = diode_state_array[j]
                        Rd = Rdf * diode_state + Rdb * (1 - diode_state)
                        statedb.append((switch_addr, diode_addr, Rsw, Rd))
                else:
                    statedb.append((switch_addr, 0, Rsw, None))
        else:
            if isinstance(i_diode_state, np.ndarray):
                for j , diode_addr in enumerate(i_diode_state):
                    diode_addr = int(diode_addr)  # get rid of np.int64
                    diode_state = diode_state_array[j]
                    Rd = Rdf * diode_state + Rdb * (1 - diode_state)
                    statedb.append((0, diode_addr, None, Rd))
            else:
                statedb.append((None, 0, None, 0))
        return statedb
    
    def create_matrix(self):
        num_switch_states = 2 ** self.num_switches
        num_diode_states = 2 ** self.num_diodes

        self.Ashape = self.expanded_statespace.A.shape
        self.Bshape = self.expanded_statespace.B.shape
        self.Cshape = self.expanded_statespace.C.shape
        self.Dshape = self.expanded_statespace.D.shape

        states = (num_switch_states, num_diode_states)

        self.A = np.zeros(states + self.Ashape)
        self.B = np.zeros(states + self.Bshape)
        self.C = np.zeros(states + self.Cshape)
        self.D = np.zeros(states + self.Dshape)

        for state in self.enum_switch_states():
            sw = state[0]
            d = state[1]
            RDB = {}
            for i in range(self.num_switches):
                RDB[self.switch_R[i]] = state[2][i]
            for i in range(self.num_diodes):
                RDB[self.diode_R[i]] = state[3][i]
            print(self.switch_db)
            print(self.diode_R)
            print(self.switch_R)
            print(RDB)
            subcircuit = self.expanded_circuit.subs(RDB)
            substate = subcircuit.state_space()
            self.A[sw][d] = np.array(substate.A.tolist(), dtype=float)
            self.B[sw][d] = np.array(substate.B.tolist(), dtype=float)
            self.C[sw][d] = np.array(substate.C.tolist(), dtype=float)
            self.D[sw][d] = np.array(substate.D.tolist(), dtype=float)
            print(". ", end="")

    def sim(self):
        input = np.array([[0.0], [0.7]])
        t = np.linspace(0, 0.06, 57)
        dt = 0.06 / 200
        #x = np.array([np.zeros(len(ss.A.tolist()[0]))], dtype=float)

        # xdot = A @ x + B @ input
        # print(type(x), type(xdot * dt) )
        # x += xdot * dt
        # y = C @ x + D @ input
        Vin = 2.5 * np.sin(6.28 * 50 * t)
        Id = np.zeros(len(Vin))
        Vd = np.zeros(len(Vin))
        k = self.diode_I_V_output_indices[0]
        for i in range(len(Vin)):
            Vi = Vin[i]
            V = input[1][0]
            input[0][0] = Vi
            output = np.array(ss.D.tolist()) @ input
            Vdi = output[k][0] * R_D0 + 0.7
            Vd[i] = Vdi
            if output[k][0] < 0.7 / 1e6 / 1000:
                input[1][0] = 0.7
                ct = self.expanded_circuit.subs({"R_D0": 1e6})
                ss = ct.state_space()
                R_D0 = 1e6
            else:
                input[1][0] = 0.7
                ct = self.expanded_circuit.subs({"R_D0": 0.01})
                ss = ct.state_space()
                R_D0 = 0.01
            if input[1][0] != Id[i-1]:
                #pass
                output = np.array(ss.D.tolist()) @ input
            Id[i] = output[k][0]
        return Vin, Id, Vd
        
    def __str__(self):
        s  = "PowerCircuit with netlist:\n"
        s += self.netlist
        s += "\nExpanded netlist:\n"
        s += self.expanded_netlist
        s += "\n\nSwitches:\n"
        s += str(list(self.switch_list))
        s += "\n\nDiodes:\n"
        s += str(list(self.diode_list))
        s += "\n\nExpanded circuit inputs:\n"
        s += str(self.expanded_inputs)
        s += "\n\nExpanded circuit outputs:\n"
        s += str(self.expanded_outputs)
        return s
        


original_netlist = """
V1 1 0 3
D0 2 0
D5 2 0
D2 2 0
D3 2 0
R1 1 2 1
C1 2 0 1e-4
#SW1 1 0
"""

pc = PowerCircuit(original_netlist)
print(pc)
import sys; sys.exit(0)
Vin, Id, Vd = pc.sim()
t = np.arange(len(Vin))

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_title("Diode test")
ax.set(xlabel="tVin")
ax.set(ylabel="Idiode")
ax.plot(t, Vin, linewidth=1)
ax.plot(t, Id, linewidth=1, color="tab:green")
#ax.plot(t, Vd, linewidth=1, color="tab:red")
#ax.legend()



# Define the diode parameters (e.g., for the ON state)
# Note: In Lcapy, one can pass parameters as a dict to analysis methods
diode_params = {
    'R_D1': 0.1,  # Low resistance when ON
    'V_D1': 0.7   # Forward voltage drop
}



# # Diode turns ON if above 0.7V, but stays ON until below 0.6V
# # current_states is the boolean array from the PREVIOUS timestep
# is_high = y[diode_indices] > 0.7
# is_low = y[diode_indices] < 0.6

# # Vectorized update logic
# # If is_high is True, state becomes True. 
# # If is_low is True, state becomes False.
# # If neither, state stays what it was.
# new_states = np.where(is_high, True, np.where(is_low, False, current_states))


# # 1. Check Diodes (from output vector y)
# d_on = y[diode_indices] > 0.7 

# # 2. Check Switches (e.g., a 50Hz PWM signal)
# s1_on = True if (t % 0.02) < 0.01 else False
# s2_on = some_other_logic

# # 3. Concatenate into one boolean array
# # Order: [D1, D2, S1, S2]
# all_states = np.append(d_on, [s1_on, s2_on])

# # 4. Get Matrix Index
# idx = all_states @ weights # weights = [1, 2, 4, 8]
# Ak = All_A_matrices[idx]

# from itertools import product
# import numpy as np

