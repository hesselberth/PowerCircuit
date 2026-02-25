#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:05:58 2026

@author: Marcel Hesselberth

Library to simulate power electronic circuits comprised of sources, linear
circuit elements like resistors, capacitors and inductors and nonlinear
elements like diodes and switches. The library takes the MNA (modified nodal
analysis) from LCapy and expands it to a switching circuit.
"""


import os, re
import numpy as np
from lcapy import Circuit
from sympy import lambdify
import math


# Default switch values, SI units
Rds_on = 0.01  # SWitch on
Rds_off = 1e7  # SWitch off is modeled as a high resistance state
Vf = 0.6       # Diode voltage drop
Rdf = 0.05     # Diode forward resistance
Rdr = 1e6      # Diode reverse blocking is modeled as a resistance state


class NetlistArray(np.ndarray):
    def __new__(cls, input_array, mapping=None):
        obj = np.asanyarray(input_array).view(cls)
        # mapping format: { 'name': (pos_idx, neg_idx_or_None) }
        obj.mapping = mapping if mapping is not None else {}
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.mapping = getattr(obj, 'mapping', {})

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self.mapping:
                raise KeyError(f"Key '{key}' not found in netlist mapping.")
            
            m = self.mapping[key]
            if isinstance(m, int):
                pos, neg = self.mapping[key], None
            else:
                pos, neg = self.mapping[key]
            
            # Access base ndarray to ensure a standard np.ndarray is returned
            base = self.view(np.ndarray)
            
            if neg is None:
                return base[pos]
            else:
                # Vectorized subtraction across all columns (time steps)
                return base[pos] - base[neg]
        
        # Standard slicing returns a ndarray view
        return super().__getitem__(key).view(np.ndarray)


prefixes = {
    12: 'T',
     9: 'G',
     6: 'M',
     3: 'k',
     0: '',
    -3: 'm',
    -6: 'u',
    -9: 'n',
    -12: 'p',
    -15: 'f'
}


def pretty_prefix(value_str):
    val = float(value_str)
    if val == 0: return "0"
    exponent = math.floor(math.log10(val))
    eng_exponent = (exponent // 3) * 3
    scaled_val = val / (10**eng_exponent)
    prefix = prefixes.get(eng_exponent, 'e' + str(eng_exponent))
    return f"{scaled_val:g}\,{prefix}"


class PowerCircuit:
    """
    Class implementing a nonlinear circuit containing switches and diodes.
    Because of the nonlinear nature of these circuits algebraic analysis is
    generally impossible. To increase computational performance, all sympy
    matrices are therefore converted to numpy. This means that no component
    values may be left unspecified.
    """
    def __init__(self, netlist, **kwargs):
        """Constructor.
        The main argument is a string containing a netlist
        """
        self.netlist = netlist

        self.Rds_on = Rds_on
        self.Rds_off = Rds_off
        self.Vf = Vf
        self.Rdf = Rdf
        self.Rdr = Rdr

        self.expanded_netlist, self.draw_netlist, self.switch_db, self.diode_db = \
            self.process_netlist(netlist)
        print(self.switch_db)
        print(self.diode_db)
        print("expanded")
        print(self.expanded_netlist)
        print()
        print(self.draw_netlist)
        #import sys
        #sys.exit(0)
        self.expanded_circuit = Circuit(self.expanded_netlist)
        self.expanded_statespace = self.expanded_circuit.state_space()
        
        self.sym_A = self.expanded_statespace.A
        self.sym_B = self.expanded_statespace.B
        self.sym_C = self.expanded_statespace.C
        self.sym_D = self.expanded_statespace.D
        
        self.switch_list = list(self.switch_db.keys())
        self.diode_list = list(self.diode_db.keys())
        
        self.switch_R = [self.switch_db[switch] for switch in self.switch_list]
        self.diode_V = [self.diode_db[diode]["V"] for diode in self.diode_list]
        self.diode_R = [self.diode_db[diode]["R"] for diode in self.diode_list]
        self.diode_int = [self.diode_db[diode]["node"] for diode in self.diode_list]

        self.expanded_inputs = [str(inp) for inp in self.expanded_statespace.u]        
        self.expanded_outputs = [str(outp)[:-3] for outp in self.expanded_statespace.y]
        self.num_expanded_inputs = len(self.expanded_inputs)
        self.num_expanded_outputs = len(self.expanded_outputs)

        self.u = [inp for inp in self.expanded_inputs if str(inp) not in self.diode_V]
        self.u = np.array([self.u]).T
        self.s = list(self.switch_list)
        self.d = list(self.diode_list)
        
        try:
            self.x = [str(x)[:-3] for x in self.expanded_statespace.x.tolist()[0]]
            self.x = np.array([self.x]).T
        except:
            self.x = None
        self.y = list(self.ydict.keys())
        self.output_items = self.ydict.items()

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

        self.symbols = self.switch_R + self.diode_R
        self.lambd_A = lambdify(self.symbols, self.sym_A, 'numpy')
        self.lambd_B = lambdify(self.symbols, self.sym_B, 'numpy')
        self.lambd_C = lambdify(self.symbols, self.sym_C, 'numpy')
        self.lambd_D = lambdify(self.symbols, self.sym_D, 'numpy')
        self.mkcache()


    def process_netlist(self, netlist_str):
        sim_lines = []
        draw_lines = []
        
        # Databases voor de PowerCircuit klasse
        switches = {} # { 'SW1': 'R_SW1' }
        diodes = {}   # { 'D1': {'v': 'V_D1', 'r': 'R_D1', 'int': 'int_D1'} }
        
        # Regex voor componenten en metadata
        pattern = r'^(\w+)\s+(\w+)\s+(\w+)(?:\s+([^;]+))?(?:\s*;\s*(.*))?$'
        layout_keywords = ('up', 'down', 'left', 'right', 'rotate', 'angle', 'size', 'at', 'color')
    
        for line in netlist_str.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                sim_lines.append(line); draw_lines.append(line); continue
                
            match = re.match(pattern, line)
            if not match:
                sim_lines.append(line); draw_lines.append(line); continue
                
            name, n1, n2, args, meta = match.groups()
            ctype = name[0].upper()
            arg_list = args.split() if args else []
    
            # Filter layout voor de gesplitste componenten
            m_parts = [p.strip() for p in meta.split(',')] if meta else []
            layout = [p for p in m_parts if any(k in p for k in layout_keywords)]
            l_str = f"; {', '.join(layout)}" if layout else ""
    
            # --- DIODES ---
            if ctype == 'D':
                # Gebruik symbolen tussen { } voor Lcapy
                v_sym, r_sym = f"V_{name}", f"R_{name}"
                int_node = f"int_{name}"
                
                # Sim Netlist: Vf + Rf in serie
                sim_lines.append(f"{v_sym} {n1} {int_node} {{{v_sym}}} {l_str}")
                sim_lines.append(f"{r_sym} {int_node} {n2} {{{r_sym}}} {l_str}")
                
                diodes[name] = {'V': v_sym, 'R': r_sym, 'node': int_node}
                
                # Draw Netlist: Originele diode met rijke annotatie
                v_val = arg_list[0] if len(arg_list) > 0 else "0.7V"
                r_val = arg_list[1] if len(arg_list) > 1 else "10m"
                draw_lines.append(f"{name} {n1} {n2}; l={ctype}_{name[1:]}, {meta}")
    
            # --- SCHAKELAARS ---
            elif name.upper().startswith('SW'):
                r_sym = f"R_{name}"
                sim_lines.append(f"{r_name} {n1} {n2} {{{r_name}}} {l_str}")
                switches[name] = r_name
                draw_lines.append(line) # Schakelaar blijft visueel een schakelaar
    
            # --- CONDENSATOREN / SPOELEN (met optionele ESR) ---
            elif ctype in ('C', 'L', 'V') and len(arg_list) >= 2:
                val, esr = arg_list[0], arg_list[1]
                r_esr = f"R_{name}"
                int_node = f"int_{name}"
                
                sim_lines.append(f"{name} {n1} {int_node} {val} {l_str}")
                sim_lines.append(f"{r_esr} {int_node} {n2} {{{esr}}} {l_str}")
                
                # Voor draw: clean de eenheid (1uF -> 1u) om Lcapy-crashes te voorkomen
                safe_val = re.sub(r'([0-9\.]+)[a-zA-Z]+$', r'\1', val)
                pretty_val = pretty_prefix(safe_val)
                print(pretty_val)
                draw_lines.append(fr"{name} {n1} {n2} {safe_val}; l={{{ctype}_{name[1:]}{{=}}\mathrm{{{pretty_val}}}}}, a={{ {esr}\, \Omega}}, {meta}")
            elif name.upper() == "GND":
                draw_lines.append(f"W_gndsink {n1} {n2}; ground, size=0.0, {meta}")
            else:
                sim_lines.append(line)
                draw_lines.append(line)
    
        return "\n".join(sim_lines), "\n".join(draw_lines), switches, diodes


    def draw(self, *args, **kwargs):
        nodepat = r"(\S+)\s+(\S+)\s+([^;\s]+)\s*(.*)"
        lines = self.draw_netlist.strip().split('\n')
        draw_netlist = []
        nodes = [n[2:] for n in self.y if n.startswith("v_")]
        replace = {'0': '0'}
        def replace_node(n):
            v = "v_" + n
            if v in self.y or "_" in n:
                return n
            elif n in replace:
                return replace[n]
            else:
                new = "_" + n
                while "_" + new in self.y:
                    new = "_" + new
                replace[n] = new
                print(replace)
                return new
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                pass  # comment
            else:
                match = re.match(nodepat, line)
                if match:
                    part, np, nn, rem = match.groups()
                    np = replace_node(np)
                    nn = replace_node(nn)
                    if part.lower() == "gnd":
                        part = "W_gndsink"
                        rem = rem + ", ground, size=0.0"
                    s = " ".join([part, np, nn, rem])
                else:
                    s = line
            draw_netlist.append(s)
        draw_netlist = "\n".join(draw_netlist)
#        print("draw_netlist")
#        print(draw_netlist)
        cct = Circuit(draw_netlist)
        return cct.draw(*args, **kwargs)


    def mkcache(self):
        """
        Creates an array of an array of matrices A, B, C and D of the
        correct size. A[sw_arrd][d_addrr] will retrieve the A matrix for a
        switch and diode state. sw_addr follows from the dot product of 
        the switch state array with self.switch_addr .
        This procedure also creates the abcd_cache matrix cache so that
        each possible topology is computed maximum once.
        
        Also creates Ab and Bb matrices for backward Euler integration
        and Ab_m and Bb_m matrices for times-m microstepping. These matrices
        depend on integration time dt. As long as dt and m are not changed,
        those matrices are cached as well, using be_cache as a hit indicator.
        """
    
        num_switch_states = 2 ** self.num_switches
        num_diode_states = 2 ** self.num_diodes
        num_states = num_switch_states * num_diode_states

        self.Ashape = self.expanded_statespace.A.shape
        self.Bshape = self.expanded_statespace.B.shape
        self.Cshape = self.expanded_statespace.C.shape
        self.Dshape = self.expanded_statespace.D.shape

        states = (num_switch_states, num_diode_states)

        self.A = np.zeros(states + self.Ashape, dtype = float)
        self.B = np.zeros(states + self.Bshape, dtype = float)
        self.C = np.zeros(states + self.Cshape, dtype = float)
        self.D = np.zeros(states + self.Dshape, dtype = float)
        
        self.Ab = np.zeros(states + self.Ashape, dtype = float)
        self.Bb = np.zeros(states + self.Bshape, dtype = float)
        self.Ab_m = np.zeros(states + self.Ashape, dtype = float)
        self.Bb_m = np.zeros(states + self.Bshape, dtype = float)
        
        self.abcd_cache = np.zeros(states, dtype = bool)
        self.flush_be_cache(0)


    def abcd(self, sw_addr, d_addr, sw_array, d_array):
        # The assignment of values to a specific switch or diode happens here.
        # self.switch_R[i] is the resistor that implements the switch.
        # The order is identical that of self.switch_list / self.diode_list.
        RDB = {}
        for i in range(self.num_switches):
            on = sw_array[i]
            RDB[self.switch_R[i]] = self.Rds_on if on else self.Rds_off
        for i in range(self.num_diodes):
            on = d_array[i]
            RDB[self.diode_R[i]] = self.Rdf if on else self.Rdr
            RDB[self.diode_V[i]] = self.Vf #* on
        args = [RDB[s] for s in self.symbols]
        if self.x:
            self.A[sw_addr][d_addr] = self.lambd_A(*args)
            self.B[sw_addr][d_addr] = self.lambd_B(*args)
            self.C[sw_addr][d_addr] = self.lambd_C(*args)
        self.D[sw_addr][d_addr] = self.lambd_D(*args)
        self.abcd_cache[sw_addr][d_addr] = True


    def be(self, sw_addr, d_addr, sw_array, d_array, m, dt):
        if not self.abcd_cache[sw_addr][d_addr]:
            self.abcd(sw_addr, d_addr, sw_array, d_array)
        if m != self.m_cache:
            raise ValueError("cache was set up fot different m value " 
                             f"({self.m_cache}), flush it first")
        dt_micro = dt / m  # dt for microstepping
        A = self.A[sw_addr][d_addr]
        B = self.B[sw_addr][d_addr]
        I = np.eye(self.Ashape[0])
        
        if self.x:
            print("calc")
            self.Ab[sw_addr][d_addr] = np.linalg.inv(I - dt * A)
            self.Bb[sw_addr][d_addr] = self.Ab[sw_addr][d_addr] @ (dt * B)
            self.Ab_m[sw_addr][d_addr] = np.linalg.inv(I - dt_micro * A)
            self.Bb_m[sw_addr][d_addr] = self.Ab_m[sw_addr][d_addr] @ (dt_micro * B)
        self.be_cache[sw_addr][d_addr] = True


    def flush_be_cache(self, m):
           states = (2 ** self.num_switches, 2 ** self.num_diodes)
           self.be_cache = np.zeros(states, dtype = bool)
           self.dt_cache = 0
           self.m_cache = m


    @property
    def ydict(self):
        print(self.expanded_outputs)
        print(self.diode_db)
        d = {name : i for i, name in enumerate(self.expanded_outputs)}
        for d_name in self.diode_db:
            d_info = self.diode_db[d_name]
            del d["v_" + d_info["node"]]
            del d["i_" + d_info["V"]]
            d["i_"+d_name] = d["i_" + d_info["R"]]
            del d["i_" + d_info["R"]]
        for sw_name in self.switch_db:
            R = self.switch_db[sw_name]
            d["i_"+sw_name] = d["i_" + R]
            del d["i_" + R]
        s = {key: d[key] for key in sorted(d, key=d.get)}  # ordered by key
        return s
    
    
    def extended_ydict(self):
        ydict = self.ydict
        result = ydict.copy()
        for item in self.ydict:
            print(item)
        print(self.ydict)


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


    def sim_step(self, sw_addr, d_addr, sw_array, d_array, u_exp, x, m, t, dt, lc, result):
        x_prev = x
        d_array_prev = d_array
        dt_micro = dt / m
        
        # Try a dt step
        u = u_exp[:, [t]]
        if not self.be_cache[sw_addr][d_addr]:
            self.be(sw_addr, d_addr, sw_array, d_array, m, dt) # Loads into self.Ab, self.Bb
        Ab = self.Ab[sw_addr][d_addr]
        Bb = self.Bb[sw_addr][d_addr]
        C = self.C[sw_addr][d_addr]
        D = self.D[sw_addr][d_addr]
        
        # Trial integration
        if lc:
            x_trial = (Ab @ x_prev) + (Bb @ u)
            y_trial = C @ x_trial + D @ u
        else:
            y_trial = D @ u
                
        I_diodes = y_trial[self.diode_I_V_output_indices].ravel()
        d_array_trial = (I_diodes > 1e-10) * 1  # 1e-10: supress numerical noise
        
        # Check for diode commutation
        if (d_array_trial == d_array_prev).all():
            # No switching, accept the macro step
            # print("tn", t, u[0])
            x = x_trial
            y = y_trial
            d_array = d_array_trial
        else:
            # Commutation detected, backtrack.
            # Prepare micro-interpolation for u
            # print("commutation discovered at", t)
            utm1 = u_exp[:, [t-1]] if t > 0 else u_exp[:, [t]]
            ut = u_exp[:, [t]] 
            du = ut - utm1
            # (arange(1, m+1) / m to end up at end of macro interval
            u_interpolated = utm1 + (np.arange(1, m + 1 ) / m) * du
            # t_micro = t - 1 + np.arange(1, m+1 ) / (m)
            # Microstepping loop
            for i in range(m):
                x_prev = x  # first pass from function argument
                assert(m == self.m_cache)
                u_micro = u_interpolated[:, [i]]
                #t_i = 0.00006 * t_micro[i]
                #v_i = 12 * np.sin(6.28 * 50 * t_i)
                #u_micro[0] = v_i
                # print("tc", t-1 + ((i+1) / m), t_micro[i], u_micro[0], v_i , u_micro[0]- v_i)

                # Fetch micro matrices for interval dt/m, m>1
                if not self.be_cache[sw_addr][d_addr]:
                    self.be(sw_addr, d_addr, sw_array, d_array, m, dt) 
                Abm = self.Ab_m[sw_addr][d_addr]
                Bbm = self.Bb_m[sw_addr][d_addr]
                C = self.C[sw_addr][d_addr]
                D = self.D[sw_addr][d_addr]
                
                #Integrate microstep
                if lc:
                    x = (Abm @ x_prev) + (Bbm @ u_micro)
                    y = C @ x + D @ u_micro
                else:
                    y = D @ u_micro
                
                # Check commutation within microstep
                I_diodes = y[self.diode_I_V_output_indices].T[0]
                d_array = (I_diodes > 0) * 1
                
                if (d_array != d_array_prev).any():
                    # Update topology mid-micro-loop
                    #print("commutation", d_array, t_micro[i] )
                    d_addr = int(np.dot(d_array, self.diode_addr))
                    d_array_prev = d_array
                    # Re-run this microstep with new topology
                    if not self.be_cache[sw_addr][d_addr]:
                        self.be(sw_addr, d_addr, sw_array, d_array, m, dt) 
                    Abm = self.Ab_m[sw_addr][d_addr]
                    Bbm = self.Bb_m[sw_addr][d_addr]
                    C = self.C[sw_addr][d_addr]
                    D = self.D[sw_addr][d_addr]
                    if lc:
                        x = (Abm @ x_prev) + (Bbm @ u_micro)
                        y = C @ x + D @ u_micro
                    else:
                        y = D @ u_micro  
        result[:, [t]] = y
        return sw_addr, d_addr, d_array, x


    def sim(self, u, s, n, dt, m = 1, x=None):
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
        
        # if user passed row vector or list, fix it
        if isinstance(u, list):
            u = np.array(u)
        if u.shape == (1, self.num_inputs):
            col_vector = np.array(u).reshape(-1, 1)
            u = np.tile(col_vector, (1, n))
            assert(u.shape == (self.num_inputs, n))
        if u.shape == (self.num_inputs, n):
            u_exp = np.ones((self.num_expanded_inputs, n)) * self.Vf
            u_exp[self.expanded_input_indices] = u
        else:
            raise ValueError(f"Input array must have shape (m, 1) or (m, n) " 
                             f"where m is the number of inputs (got {u.shape})")

        if len(s) == self.num_switches:  # Values must be bool or 0/1
            sw_array = np.array(s, dtype=bool) * 1
            sw_addr = int(np.dot(sw_array, self.switch_addr))
        else:
            raise ValueError(f"Circuit has {self.num_switches} switches, "
                             f"got {len(s)}.")

        if x != None:
            if x.shape != (self.Ashape[0], 1):
                raise ValueError(f"x should have shape {self.Ashape}, "
                                 f"(got {x.shape}.")
        else:
            x = np.zeros((self.Ashape[0], 1))
            

        if dt <=0:
            raise ValueError("timestep dt <= 0")

        if not isinstance(m, int):
            raise TypeError(f"m must be integer (got {type(m)})")
        if m < 1:
            raise ValueError(f"m must be positive (got {m})")
        
        if self.dt_cache != dt or self.m_cache != m:
            self.flush_be_cache(m)

        lc = (x.shape != (0, 0))  # self.x
        print("lc", lc, x.shape)
        d_addr = 0
        d_array = np.zeros(self.num_diodes, dtype=int)
        output = np.empty((self.num_expanded_outputs, n), dtype = float)
        
        for i in range(n):
            sw_addr, d_addr, d_array, x = self.sim_step(sw_addr, d_addr, sw_array, d_array, u_exp, x, m, i, dt, lc, output)

        return NetlistArray(output, mapping = self.ydict)
        #return {k: output[i] for k, i in self.output_items}


    def __str__(self):
        s  = "PowerCircuit with netlist:\n"
        s += self.netlist
        #s += "\nExpanded netlist:\n"
        #s += self.expanded_netlist
        s += "\nSwitches: "
        s += str(list(self.switch_list))
        s += "\nDiodes: "
        s += str(list(self.diode_list))
        #s += "\n\nExpanded circuit inputs:\n"
        #s += str(self.expanded_inputs)
        #s += "\n\nExpanded circuit outputs:\n"
        #s += str(self.expanded_outputs)
        s += "\nstate vector: "
        s += str(self.x)        
        s += "\ninputs: "
        s += str(self.u)
        s += "\noutputs: "
        s += str(self.y)
        return s


netlist_fb = """
V1 1 3 
D0 0 1
D1 1 2
D2 0 3
D3 3 2
R1 2 0 10
C1 2 0 1000e-6
"""

netlist_d = """
V1 1 0
V2 5 0 7
D0 1 2
R1 2 0 100
R2 2 3 0.001
C1 3 0 200e-6
SW1 2 0
"""

netlist_d = """
Vin 1 0_1; down
D0 1 2; right, size=1.5
R2 2 3 0.1; down, b=100
C1 3 0 1000e-6 .1; down
W1 2 4; right
R1 4 0_4 100; down
W4 0 0_4; right, size=1.5
W5 0_1 0; right
gnd 0 0_g; down
; style=american
"""

pc = PowerCircuit(netlist_d)

print("exp inp:")
print(pc.expanded_inputs)
print(pc.num_inputs)

pc.draw("circuit.png")

pc.draw()

import sys
#sys.exit(0)

dt = 0.00006
i = np.arange(1000)
t = i * dt

Vin = 12 * np.sin(6.28 * 50 * t)
n = len(Vin)
u = np.array([Vin])

y = pc.sim(u, [], n, dt, 5)


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


fig, ax = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 1]})

axf = ax[0]
axs = ax[1]

axf.set_title("Diode rectifier example")
axf.set(xlabel="t")
axf.set(ylabel="V(V)")
ax2 = axf.twinx()
ax2.set(ylabel="I(A)")
axf.plot(t, y["v_1"] - 0*y["v_3"], linewidth=1, label="Vin")
ax2.plot(t, y["i_D0"], linewidth=1, label="Idiode", color="tab:green")
axf.plot(t, y['v_2'], linewidth=1, label="Vout", color="tab:red")
axf.legend(loc="lower left")
ax2.legend()

img = mpimg.imread('circuit.png')
axs.imshow(img)
axs.axis('off')

plt.tight_layout()


print("u symbols:", [str(u) for u in pc.expanded_statespace.u])
print("Does V_D0 appear in u? →", any('V_D0' in str(u) for u in pc.expanded_statespace.u))
print("y contains i_V_D0?", any('i_V_D0' in str(y) for y in pc.expanded_statespace.y))
print("y contains i_R_D0?", any('i_R_D0' in str(y) for y in pc.expanded_statespace.y))

plt.show()



