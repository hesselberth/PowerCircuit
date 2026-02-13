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
Rdr = 1e6      # Diode reverse blocking is modeled as a resistance state


"""
Class implementing a nonlinear circuit containing switches and diodes.
Because of the nonlinear nature of these circuits algebraic analysis is
generally impossible. To increase computational performance, all sympy
matrices are therefore converted to numpy. This means that no component
values may be left unspecified.
"""
class PowerCircuit:
    def __init__(self, netlist, **kwargs):
        self.netlist = netlist

        self.Rds_on = Rds_on
        self.Rds_off = Rds_off
        self.Vf = Vf
        self.Rdf = Rdf
        self.Rdr = Rdr

        self.expanded_netlist, self.switch_db, self.diode_db = \
            self.expand_netlist(netlist)
        self.expanded_circuit = Circuit(self.expanded_netlist)
        self.expanded_statespace = self.expanded_circuit.state_space()
        
        self.switch_list = self.switch_db.keys()
        self.diode_list = self.diode_db.keys()
        self.switch_R = [self.switch_db[switch] for switch in self.switch_list]
        self.diode_V = [self.diode_db[diode][0] for diode in self.diode_list]
        self.diode_R = [self.diode_db[diode][1] for diode in self.diode_list]
        self.diode_int = [self.diode_db[diode][2] for diode in self.diode_list]
        self.expanded_inputs = [str(inp) for inp in self.expanded_statespace.u]
        
        self.expanded_outputs = [str(outp)[:-3] for outp in self.expanded_statespace.y]
        self.num_expanded_inputs = len(self.expanded_inputs)
        self.num_expanded_outputs = len(self.expanded_outputs)

        self.u = [inp for inp in self.expanded_inputs if str(inp) not in self.diode_V]
        self.u = np.array([self.u]).T
        self.s = list(self.switch_list)
        try:
            self.x = [str(x)[:-3] for x in self.expanded_statespace.x.tolist()[0]]
            self.x = np.array([self.x]).T
        except:
            self.x = None

        self.y = self.expanded_outputs
        self.output_items = self.ydict().items()

        self.expanded_input_indices = [self.expanded_inputs.index(inp) for inp in self.u]
        self.diode_V_input_indices = [self.expanded_inputs.index(Vstr) for Vstr in self.diode_V]
        self.diode_I_V_output_indices = [self.expanded_outputs.index("i_"+outp) for outp in self.diode_V]
        self.diode_I_R_output_indices = [self.expanded_outputs.index("i_"+outp) for outp in self.diode_R]
        self.diode_v_int_output_indices = [self.expanded_outputs.index("v_"+outp) for outp in self.diode_int]

        self.input_indices = [inp for inp in self.expanded_inputs if str(inp) not in self.diode_V]

        self.num_inputs = self.u.shape[0]
        self.num_switches = len(self.switch_list)
        self.num_diodes = len(self.diode_list)

        self.switch_addr = 2 ** np.arange(self.num_switches, dtype = int)[::-1]
        self.diode_addr = 2 ** np.arange(self.num_diodes, dtype = int)[::-1]
        
        self.create_ABCD()


    def expand_netlist(self, netlist_str):
        """
        Replaces 'Dname n1 n2' with a series R and V source and
        SWname n1 n2 by R.
        """
        lines = netlist_str.strip().split('\n')
        new_netlist = []
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
                new_netlist.append(f"V_{name} {int_node} {cathode} {{V_{name}}}")
                diodes[name] = (f"V_{name}", f"R_{name}", f"int_{name}")
            elif parts[0].startswith('SW') and len(parts) >= 3:
                name = parts[0]           # e.g., SW1
                Np = parts[1]             # e.g., 2
                Nm = parts[2]             # e.g., 3
                
                # Create the switch resistance
                new_netlist.append(f"R_{name} {Np} {Nm} {{R_{name}}}")
                switches[name] = (f"R_{name}")
            else:
                new_netlist.append(line)
        return '\n'.join(new_netlist), switches, diodes


    def create_ABCD(self):
        """
        Creates an array of an array of matrices A, B, C and D of the
        correct size. A[sw_arrd][d_addrr] will retrieve the A matrix for a
        switch and diode state. sw_addr follows from the dot product of 
        the switch state array with self.switch_addr .
        This procedure also creates the mcache matrix cache so that each
        matrix is computed maximum 1 time.
        """
    
        num_switch_states = 2 ** self.num_switches
        num_diode_states = 2 ** self.num_diodes
        num_states = num_switch_states * num_diode_states

        self.Ashape = self.expanded_statespace.A.shape
        self.Bshape = self.expanded_statespace.B.shape
        self.Cshape = self.expanded_statespace.C.shape
        self.Dshape = self.expanded_statespace.D.shape

        states = (num_switch_states, num_diode_states)

        self.A = np.zeros(states + self.Ashape)
        self.B = np.zeros(states + self.Bshape)
        self.C = np.zeros(states + self.Cshape)
        self.D = np.zeros(states + self.Dshape)
        
        self.mcache = np.zeros(states, dtype = bool)


    def compute_matrix(self, sw_addr, d_addr, sw_array, d_array):
        # The assignment of values to a specific switch or diode happens here.
        # self.switch_R[i] is the resistor that implements the switch.
        # The order is identical to self.switch_list / self.diode_list.

        if self.mcache[sw_addr][d_addr]:
            A = self.A[sw_addr][d_addr]
            B = self.B[sw_addr][d_addr]
            C = self.C[sw_addr][d_addr]
            D = self.D[sw_addr][d_addr]
            return A, B, C, D

        #sw_array = self.array_from_addr(sw_addr, self.num_switches)
        #d_array = self.array_from_addr(d_addr, self.num_diodes)
        RDB = {}
        for i in range(self.num_switches):
            switch_state = sw_array[i]
            Rsw = self.Rds_on * switch_state + self.Rds_off * (1 - switch_state)
            RDB[self.switch_R[i]] = Rsw
        for i in range(self.num_diodes):
            switch_state = d_array[i]
            Rsw = self.Rdf * switch_state + self.Rdr * (1 - switch_state)
            RDB[self.diode_R[i]] = Rsw
        subcircuit = self.expanded_circuit.subs(RDB)
        substate = subcircuit.state_space()
        try:
            A = self.A[sw_addr][d_addr] = np.array(substate.A.tolist(), dtype=float)
            B = self.B[sw_addr][d_addr] = np.array(substate.B.tolist(), dtype=float)
            C = self.C[sw_addr][d_addr] = np.array(substate.C.tolist(), dtype=float)
        except:
            A = B = C = None
            # TODO check this in sim
        D = self.D[sw_addr][d_addr] = np.array(substate.D.tolist(), dtype=float)
        self.mcache[sw_addr][d_addr] = True
        print(". ", end="")


    def ydict(self):
        d = {name : i for i, name in enumerate(self.expanded_outputs)}
        for d_name in self.diode_db:
            V, R, internal_node = self.diode_db[d_name]
            del d["v_" + internal_node]
            del d["i_" + R]
            d["i_"+d_name] = d["i_" + V]
            del d["i_" + V]
        for sw_name in self.switch_db:
            R = self.switch_db[sw_name]
            d["i_"+sw_name] = d["i_" + R]
            del d["i_" + R]
        s = {key: d[key] for key in sorted(d, key=d.get)}  # ordered by key
        return s


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
            sw_array = np.array(s, dtype=bool) * 1
            sw_addr = int(np.dot(sw_array, self.switch_addr))
        else:
            raise ValueError(f"Circuit has {self.num_switches} switches, "
                             f"got {len(s)}.")

        if x != None:
            if x.shape != (self.Ashape[0], 1):
                raise(ValueError(f"x should have shape {self.Ashape}, "
                                 f"(got {x.shape}."))
        else:
            x = np.zeros((self.Ashape[0], 1), dtype=float)

        d_addr = 0
        prev_d_addr = d_addr
        d_array = np.zeros(2 ** self.num_diodes)

        if not self.mcache[sw_addr][d_addr]:
            self.compute_matrix(sw_addr, d_addr, sw_array, d_array)
        A = self.A[sw_addr][d_addr]
        B = self.B[sw_addr][d_addr]
        C = self.C[sw_addr][d_addr]
        D = self.D[sw_addr][d_addr]

        output = np.empty((self.num_expanded_outputs, n), dtype = float)
        
        for i in range(n):
            if not fixed_input:
                u = u_exp[:, [i]]

            if self.x:
                xdot = A @ x + B @ u
                x += xdot * dt
                y = C @ x + D @ u
            else:
                y = D @ u

            I_diodes = y[self.diode_I_V_output_indices].T[0]
            forward = (I_diodes > 0) * 1
            d_addr = np.dot(forward, self.diode_addr)

            if d_addr != prev_d_addr:
                if self.x:
                    y = C @ x + D @ u
                else:
                    y = D @ u
                if not self.mcache[sw_addr][d_addr]:
                    self.compute_matrix(sw_addr, d_addr, sw_array, forward)
                A = self.A[sw_addr][d_addr]
                B = self.B[sw_addr][d_addr]
                C = self.C[sw_addr][d_addr]
                D = self.D[sw_addr][d_addr]
            prev_d_addr = d_addr
            output[:, [i]] = y

        return {k: output[i] for k, i in self.output_items}
        
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
        

netlist_fb = """
V1 1 3 0
D0 0 1
D1 1 2
D2 0 3
D3 3 2
R1 2 0 100
C1 2 0 1e-3
"""

netlist_d = """
V1 1 0
V2 5 0 7
D0 1 2
R1 2 0 100
R2 2 3 0.1
C1 3 0 1e-3
SW1 2 0
"""



pc = PowerCircuit(netlist_fb)
print(pc)

t = pc.t(60e-3, 1e-8)

t = np.linspace(0, 0.06, 10001)
dt = 0.06 / 10000

Vin = 12 * np.sin(6.28 * 50 * t)
n = len(Vin)
u = np.array([Vin])

y = pc.sim_step(u, [], n, dt)


import sys
#sys.exit(0)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_title("Diode test")
ax.set(xlabel="t")
ax.set(ylabel="V(V)")
ax2 = ax.twinx()
ax2.set(ylabel="I(A)")
ax.plot(t, y["v_1"], linewidth=1, label="Vin")
ax2.plot(t, y["i_D0"], linewidth=1, label="Idiode", color="tab:green")
ax.plot(t, y["v_2"], linewidth=1, label="Vout", color="tab:red")
#ax.legend(loc="lower left")
#ax2.legend()




# # Diode turns ON if above 0.7V, but stays ON until below 0.6V
# # current_states is the boolean array from the PREVIOUS timestep
# is_high = y[diode_indices] > 0.7
# is_low = y[diode_indices] < 0.6

