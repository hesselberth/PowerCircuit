#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:05:58 2026

@author: Marcel Hesselberth

Library to simulate power electronic circuits comprising of sources, linear
circuit elements like resistors, capacitors and inductors and nonlinear
elements like diodes and switches. The library takes the MNA (modified nodal
analysis) from LCapy and expands it to a switching circuit.
"""


import numpy as np
from lcapy import Circuit, pprint


# Default switch values, SI units
Rds_on = 0.01  # SWitch on
Rds_off = 1e7  # SWitch off is modeled as a high resistance state
Vf = 0.7       # Diode voltage drop
Rdf = 0.05     # Diode forward resistance
Rdb = 1e6      # Diode blocking is modeled as a resistance state


def bool_pow(n, prev=[]):
    """
    Boolean powerset of size n, where n is an integer.
    Creates a set of lists of boolean variables covering all distinct
    possibilities.
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
matrices are therefore converted to numpy. This means that no component
values may be left unspecified.
"""
# TODO: check that
class PowerCircuit:
    def __init__(self, netlist, **kwargs):
        self.netlist = netlist

        self.Rds_on = Rds_on
        self.Rds_off = Rds_off
        self.Vf = Vf
        self.Rdf = Rdf
        self.Rdb = Rdb

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
        self.expanded_outputs = [str(inp) for inp in self.symbolic_statespace.y]
        self.num_expanded_inputs = len(self.expanded_inputs)
        self.num_expanded_outputs = len(self.expanded_outputs)
        self.diode_V_input_indices = [self.expanded_inputs.index(Vstr) for Vstr in self.diode_V]
        print("diode_V_input_indices:")
        print(self.diode_V_input_indices)

        self.input_indices = [inp for inp in self.expanded_inputs if str(inp) not in self.diode_V]
        self.u = [inp for inp in self.expanded_inputs if str(inp) not in self.diode_V]
        self.u = np.array([self.u]).T

        print("expanded_statespace.u:")
        print(self.expanded_statespace.u)
        self.expanded_input_indices = [self.expanded_inputs.index(inp) for inp in self.u]
        print("expanded_input_indices:")
        print(self.expanded_input_indices)
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
        
        self.u = [inp for inp in self.expanded_inputs if str(inp) not in self.diode_V]
        self.u = np.array([self.u]).T
        self.s = list(self.switch_list)
        print("u", self.u)
        print(self.u.shape)
        print(self.s)
        self.num_inputs = self.u.shape[0]
        print(self.num_inputs)

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
        """
        Enumerate all possible switch states (closed and open switches,
        blocking and forward biased diodes). 

        Returns a list of 4-tuples. Each 4-tuple contains 
        """
        Rsw_on = self.Rds_on
        Rsw_off = self.Rds_off
        Rd_on = self.Rdf
        Rd_off = self.Rdb

        switch_state_list = bool_pow(self.num_switches) if self.num_switches > 0 else []
        switch_state_array = np.array(switch_state_list, dtype=int)
        i_switch_state = np.dot(switch_state_array, self.switch_addr)
        diode_state_list = bool_pow(self.num_diodes) if self.num_diodes > 0 else []
        diode_state_array = np.array(diode_state_list, dtype=int)
        i_diode_state = np.dot(diode_state_array, self.diode_addr)
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
        self.statedb = statedb
    
    def create_matrix(self):
        self.enum_switch_states()
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

        # The assignment of values to a specific switch or diode happens here.
        # self.switch_R[i] is the resistor that implements the switch.
        # The order is identical to self.switch_list / self.diode_list.
        for state in self.statedb:
            sw = state[0]
            d = state[1]
            RDB = {}
            for i in range(self.num_switches):
                RDB[self.switch_R[i]] = state[2][i]
            for i in range(self.num_diodes):
                RDB[self.diode_R[i]] = state[3][i]
            subcircuit = self.expanded_circuit.subs(RDB)
            substate = subcircuit.state_space()
            try:
                self.A[sw][d] = np.array(substate.A.tolist(), dtype=float)
                self.B[sw][d] = np.array(substate.B.tolist(), dtype=float)
                self.C[sw][d] = np.array(substate.C.tolist(), dtype=float)
            except:
                print("No reactive elements found")
                # TODO check this in sim
            self.D[sw][d] = np.array(substate.D.tolist(), dtype=float)
            print(". ", end="")

    def t(self, *args):
        if len(args) <2:
            raise ValueError("t requires 2 or 3 arguments \
                             (<tstop>, <dt>) or (<tstart>, <tstop>, <dt>)")
        elif len(args) == 2:
            tstart = 0
            tstop = args[0]
            dt = args[1]
        elif len(args) == 3:
            tstart = args[0]
            tstop = args[1]
            dt = args[2]
        else:
            raise ValueError("t requires 2 or 3 arguments \
                             (<tstop>, <dt>) or (<tstart>, <tstop>, <dt>)")
        return np.arange(tstart, tstop, dt)
        
    def sim_step(self, u, s, n, dt, x=None):
        """
        Perform a number of simulation time steps.
        dt : Time step in seconds.
        u  : Input vector. The format is according to PowerCircuit.u. There
             are 2 possibilities:
                 1. shape(u) == shape (Powercircuit.u) = (m, 1): 1 set of
                 fixed inputs, assumed constant throughout the simulation.
                 2. shape(u) == (m, n): the m inputs are specified for each
                 n time steps.
        s  : Switch configuration. Must be a sequence of length <num_switches>.
             The switch values can be True/False or 0/1.
        nsteps : Integer specifying the number of steps to simulate.
        """
        
        if u.shape == (self.num_inputs, 1):
            fixed_input = True
            u_exp = np.ones((self.num_expanded_inputs, 1)) * self.Vf
        elif u.shape == (self.num_inputs, n):
            fixed_input = False
            u_exp = np.ones((self.num_expanded_inputs, n), dtype = float) * self.Vf
            u_exp[self.expanded_input_indices] = u
        else:
            raise ValueError(f"Input array must have shape (m, 1) or (m, n) " 
                              "where m is the number of inputs.")
        if len(s) == self.num_switches:  # Values must be bool or 0/1
            sw_addr = np.dot(s, self.switch_addr)
        else:
            raise ValueError(f"Circuit has {self.num_switches} switches, "
                              "got {len(s)}.")

        if x != None:
            if x.shape != (self.Ashape[0], 1):
                raise(ValueError(f"x should have shape {self.Ashape}, "
                                  "(got {x.shape}."))
        else:
            x = np.zeros((self.Ashape[0], 1), dtype=float)
        print("A", self.Ashape, x)
        d_addr = 0
        prev_d_addr = d_addr
        if self.Ashape == (0, 0):
            LC = False
        else:
            LC = True  # L and / or C, circuit stores energy

        A = self.A[sw_addr][d_addr]
        B = self.B[sw_addr][d_addr]
        C = self.C[sw_addr][d_addr]
        D = self.D[sw_addr][d_addr]

        u = np.ones((len(self.expanded_inputs), 1)) * self.Vf

        if fixed_input:
            u = u_exp

        output = np.empty((self.num_expanded_outputs, n), dtype = float)
        
        for i in range(n):
            if not fixed_input:
                u = u_exp[:, i]

            if LC:
                xdot = A @ x + B @ u
                x += xdot * dt
                y = C @ x + D @ u
            else:
                y = D @ u

            I_diodes = y[self.diode_I_V_output_indices].T[0]
            forward = (I_diodes > 0) * 1
            d_addr = np.dot(forward, self.diode_addr)

            A = self.A[sw_addr][d_addr]
            B = self.B[sw_addr][d_addr]
            C = self.C[sw_addr][d_addr]
            D = self.D[sw_addr][d_addr]

            if d_addr != prev_d_addr:
                y = C @ x + D @ u

            print("y", y)
            output[:, i] = y

        return output
        
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
V1 1 0 5
V2 5 0 7
D0 1 2
D1 1 2
R1 2 0 100
R2 2 3 0.1
C1 3 0 1e-3
SW1 1 0
"""

pc = PowerCircuit(original_netlist)
print(pc)

t = pc.t(60e-3, 1e-8)
print(len(t))

t = np.linspace(0, 0.06, 10001)
dt = 0.06 / 10000

Vin = 12 * np.sin(6.28 * 100 * t)
n = len(Vin)
u = np.array([Vin, Vin])
print(u)
result = pc.sim_step(u, [0], n, dt)

print(result)

import sys
sys.exit(0)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_title("Diode test")
ax.set(xlabel="t")
ax.set(ylabel="V(V)")
ax2 = ax.twinx()
ax2.set(ylabel="I(A)")
ax.plot(t, Vin, linewidth=1, label="Vin")
ax2.plot(t, Id, linewidth=1, label="Idiode", color="tab:green")
ax.plot(t, Vd, linewidth=1, label="Vout", color="tab:red")
#ax.legend(loc="lower left")
#ax2.legend()







# # Diode turns ON if above 0.7V, but stays ON until below 0.6V
# # current_states is the boolean array from the PREVIOUS timestep
# is_high = y[diode_indices] > 0.7
# is_low = y[diode_indices] < 0.6

