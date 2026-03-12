#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:05:58 2026

@author: Marcel Hesselberth

Library to simulate power electronic circuits comprised of sources, linear
circuit elements like resistors, capacitors and inductors and nonlinear
elements like diodes and switches.
"""


import re
import numpy as np
from sympy import lambdify
import math
from asteval import Interpreter
import os

# Default switch values, SI units
Rds_on = 0.01  # SWitch on
Rds_off = 1e7  # SWitch off is modeled as a high resistance state
Vf = 0.6       # Diode voltage drop
Rdf = 0.05     # Diode forward resistance
Rdr = 1e6      # Diode reverse blocking is modeled as a resistance state


class Netlist:
    def __init__(self, netlist_str):
        self.components = []
        self.node_map = {'0': 0}
        self.node_names = ['0']
        self.parse(netlist_str)

    def _get_node_id(self, name):
        if name not in self.node_map:
            new_id = len(self.node_names)
            self.node_map[name] = new_id
            self.node_names.append(name)
        return self.node_map[name]

    def parse(self, netlist_str):
        lines = netlist_str.strip().split('\n')
        for line in lines:
            line = line.split('#')[0].strip()
            lh = line.split(';', maxsplit = 1)
            if len(lh) == 1:
                line = lh
                hints = None
            else:
                line, hints = lh
            if not line: continue
            parts = re.split(r'\s+', line)
            if len(parts) < 4:
                continue
            name, n1_str, n2_str, val_str = parts[:4]
            n1, n2 = self._get_node_id(n1_str), self._get_node_id(n2_str)
            self.components.append({'name': name, 'type': name[0].upper(), 
                                    'n1': n1, 'n2': n2, 'val': float(val_str)})
        self.component_map = {c['name']: i for i, c in enumerate(self.components)}

    def get_nVL(self):
        return len(self.node_names) - 1, [c['name'] for c in self.components if c['type'] in ('V', 'L')]

    # def set_values(self, value_dict):
    #     for n, v in value_dict.items():
    #         if n in self.component_map:
    #             self.components[self.component_map[n]]["val"] = float(v)


class Netlist:
    def __init__(self, netlist_str, f=50):
        self.components = []
        self.node_map = {'0': 0}
        self.node_names = ['0']
        
        # Initialiseer de veilige Python evaluator
        self.aeval = Interpreter()
        self._setup_env(f)
        
        self.parse(netlist_str)
        # Maak een snelle index voor component-opzoekingen
        self.component_map = {c['name']: i for i, c in enumerate(self.components)}

    def _setup_env(self, f):
        """Definieer PE-specifieke functies die vectorized (array) werken."""
        self.aeval.symtable.update({
            'f': f, 
            'w': 2 * np.pi * f, 
            'pi': np.pi,
            'sin': np.sin, 
            'cos': np.cos, 
            'exp': np.exp, 
            'sqrt': np.sqrt,
            'sign': np.sign,
            # Vectorized helper functies met np.where
            'step': lambda t, t_step: np.where(t >= t_step, 1.0, 0.0),
            'pwm': lambda t, freq, d: np.where((t * freq) % 1.0 < d, 1.0, 0.0),
            'ramp': lambda t, start, slope: np.maximum(0, slope * (t - start))
        })

    def _get_node_id(self, name):
        if name not in self.node_map:
            new_id = len(self.node_names)
            self.node_map[name] = new_id
            self.node_names.append(name)
        return self.node_map[name]

    def parse(self, netlist_str):
        # Regex uitleg:
        # 1. (\w+) -> Naam (V1, R1)
        # 2. ([\w\d_]+) -> Node 1 (p, 0_1)
        # 3. ([\w\d_]+) -> Node 2 (m, 0)
        # 4. ([^;]*?) -> Waarde/Functie (alles tot de ;)
        # 5. (?:\s*;.*|$) -> Optionele ; met teken-instructies of regeleinde
        pattern = re.compile(r"(\w+)\s+([\w\d_]+)\s+([\w\d_]+)\s*([^;]*?)\s*(?:\s*;.*|$)")
        
        for line in netlist_str.strip().split('\n'):
            # Verwijder comments en witruimte aan de randen
            clean_line = line.split('#')[0].strip()
            if not clean_line: 
                continue
            
            match = pattern.match(clean_line)
            if not match: 
                continue
            
            name, n1_str, n2_str, val_part = match.groups()
            n1, n2 = self._get_node_id(n1_str), self._get_node_id(n2_str)
            
            comp_type = name[0].upper()
            val_str = val_part.strip()
            
            comp = {
                'name': name, 
                'type': comp_type, 
                'n1': n1, 
                'n2': n2, 
                'val': 0.0, 
                'expr': None
            }

            # Alleen V en I ondersteunen functies; anderen verwachten een float
            if val_str:
                if comp_type in ('V', 'I'):
                    try:
                        comp['val'] = float(val_str)
                    except ValueError:
                        comp['expr'] = val_str # Sla op als expressie string
                else:
                    try:
                        comp['val'] = float(val_str)
                    except ValueError:
                        # Voor R, L, C etc. die (nog) geen waarde hebben (bijv. PWL diodes)
                        comp['val'] = 0.0
            
            self.components.append(comp)

    def update_sources(self, t_array):
        """Update dynamische bronnen voor een gehele tijd-array."""
        self.aeval.symtable['t'] = t_array
        
        for c in self.components:
            if c['expr']:
                res = self.aeval(c['expr'])
                
                if not self.aeval.errors:
                    # res is nu een np.array van dezelfde lengte als t_array
                    c['val'] = res
                else:
                    print(f"Fout in {c['name']} expressie '{c['expr']}':")
                    for err in self.aeval.errors:
                        print(f"  -> {err.msg}")
                    c['val'] = np.zeros_like(t_array)
                    self.aeval.errors = []
            else:
                # Maak van statische waarden ook een array voor consistente matrix-berekeningen
                if isinstance(c['val'], (int, float)):
                    c['val'] = np.full_like(t_array, c['val'])

    def get_nVL(self):
        return len(self.node_names) - 1, [c['name'] for c in self.components if c['type'] in ('V', 'L')]


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


class Circuit:
    def __init__(self, netlist):
        self.netlist = netlist
        self.n_nodes, self.branches = netlist.get_nVL()
        self.dim = self.n_nodes + len(self.branches)
        
        # De 'echte' variabelen in de matrix
        self.matrix_labels = [f"v_{name}" for name in netlist.node_names[1:]] + \
                             [f"i_{name}" for name in self.branches]
        
        # De 'extra' variabelen (stromen door R en C)
        self.extra_labels = [f"i_{c['name']}" for c in netlist.components if c['type'] in ('R', 'C')]
        
        # Dit is wat je plot-code verwacht:
        #self.x_labels = self.matrix_labels + self.extra_labels
        self.u_labels = [c['name'] for c in netlist.components if c['type'] in ('V', 'I')]
        #self.u = self.u_labels
        #self.x = self.matrix_labels
        #self.y = self.x_labels

    def get_matrices(self, rdb={}):
        M = np.zeros((self.dim, self.dim))
        G = np.zeros((self.dim, self.dim))
        B = np.zeros((self.dim, len(self.u_labels)))
        
        for c in self.netlist.components:
            n1, n2 = c['n1'] - 1, c['n2'] - 1
            val, name = c['val'], c['name']
            
            if c['type'] == 'C':
                self._stamp(M, n1, n2, val)
            elif c['type'] == 'R':
                # Gebruik rdb voor matrix-commutatie (diodes)
                r_val = rdb.get(name, val)
                self._stamp(G, n1, n2, 1.0/r_val)
            elif c['type'] in ('V', 'L'):
                b_idx = self.n_nodes + self.branches.index(name)
                if n1 >= 0: G[n1, b_idx] += 1
                if n2 >= 0: G[n2, b_idx] -= 1
                if n1 >= 0: G[b_idx, n1] -= 1
                if n2 >= 0: G[b_idx, n2] += 1
                if c['type'] == 'V':
                    B[b_idx, self.u_labels.index(name)] = -1
                else:
                    M[b_idx, b_idx] = val
        return M, G, B

    def _stamp(self, mat, n1, n2, val):
        if n1 >= 0: mat[n1, n1] += val
        if n2 >= 0: mat[n2, n2] += val
        if n1 >= 0 and n2 >= 0:
            mat[n1, n2] -= val; mat[n2, n1] -= val


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

        self.expanded_netlist, self.draw_netlist, self.switch_db, \
            self.diode_db = self.process_netlist(netlist)

        self.switch_list = list(self.switch_db.keys())
        self.diode_list = list(self.diode_db.keys())
        
        self.switch_R = [self.switch_db[switch] for switch in self.switch_list]
        self.diode_V = [self.diode_db[diode]["V"] for diode in self.diode_list]
        self.diode_R = [self.diode_db[diode]["R"] for diode in self.diode_list]
        self.diode_int = [self.diode_db[diode]["node"] for diode in self.diode_list]

        self.sim_netlist = Netlist(self.expanded_netlist)
        self.circuit = Circuit(self.sim_netlist)
        
        # print("u", self.expanded_statespace.u)
        # print("x", self.expanded_statespace.x)
        # print("y", self.expanded_statespace.y)
        
        self.M0, self.G0, self.B0 = self.circuit.get_matrices()

        self.sim_inputs = [str(inp) for inp in self.circuit.u_labels]
        self.sim_outputs = [str(outp) for outp in self.circuit.matrix_labels]
        self.expanded_outputs = self.sim_outputs + self.circuit.extra_labels
        self.num_sim_inputs = len(self.sim_inputs)
        self.num_sim_outputs = len(self.sim_outputs)
        self.num_expanded_outputs = len(self.expanded_outputs)
        

        self.u = [inp for inp in self.sim_inputs if str(inp) not in self.diode_V]
        self.u = np.array([self.u]).T
        self.s = list(self.switch_list)
        self.d = list(self.diode_list)
        self.x = list(self.circuit.matrix_labels)
        self.y = list(self.ydict.keys())
        #self.output_items = self.ydict.items()

        self.sim_input_indices = [self.sim_inputs.index(inp) for inp in self.u]
        self.diode_V_input_indices = [self.sim_inputs.index(Vstr) for Vstr in self.diode_V]
        self.diode_I_V_output_indices = [self.sim_outputs.index("i_"+outp) for outp in self.diode_V]
        #self.diode_I_R_output_indices = [self.expanded_outputs.index("i_"+outp) for outp in self.diode_R]
        self.diode_v_int_output_indices = [self.sim_outputs.index("v_"+outp) for outp in self.diode_int]
        self.input_indices = [inp for inp in self.sim_inputs if str(inp) not in self.diode_V]

        self.num_inputs = self.u.shape[0]
        self.num_switches = len(self.switch_list)
        self.num_diodes = len(self.diode_list)

        self.switch_addr = 2 ** np.arange(self.num_switches, dtype = int)[::-1]
        self.diode_addr = 2 ** np.arange(self.num_diodes, dtype = int)[::-1]

        self.symbols = self.switch_R + self.diode_R

        self.mkcache()


    def resolve_wire_nodes(self, netlist_str):
        # 1. Union-Find op alle draden (W)
        parent = {}
        def find(n):
            parent.setdefault(n, n)
            if parent[n] == n: return n
            parent[n] = find(parent[n])
            return parent[n]
    
        def union(n1, n2):
            r1, r2 = find(n1), find(n2)
            if r1 != r2:
                # Hier kies je de prioriteit: 
                # Een naam zonder '_' wint altijd van een naam met '_'
                if '_' in r1 and '_' not in r2: parent[r1] = r2
                elif '_' not in r1 and '_' in r2: parent[r2] = r1
                else: parent[r1] = r2 # Alfabetisch/Willekeurig bij gelijke status
    
        # Scan eerst alle draden om groepen te maken
        lines = netlist_str.strip().split('\n')
        for line in lines:
            if line.strip().upper().startswith('W'):
                # Simpele split is hier ok, nodes staan vooraan
                parts = line.split()
                if len(parts) >= 3:
                    union(parts[1], parts[2])
    
        # 2. Maak een mapping tabel (voor later gebruik in de simulator output)
        self.node_alias_map = {n: find(n) for n in parent}
        
        return self.node_alias_map


    def resolve_wire_nodes(self, netlist_str):
        # 1. Union-Find setup
        parent = {}
        def find(n):
            parent.setdefault(n, n)
            if parent[n] == n: return n
            parent[n] = find(parent[n])
            return parent[n]
    
        def union(n1, n2):
            r1, r2 = find(n1), find(n2)
            if r1 != r2:
                # Prioriteit: namen zonder '_' winnen
                if '_' in r1 and '_' not in r2: parent[r1] = r2
                elif '_' not in r1 and '_' in r2: parent[r2] = r1
                else: parent[r1] = r2 # Alfabetisch bij gelijke status
    
        # 2. Identificeer alle nodes en verwerk de draden (W)
        pattern = r'^(\w+)\s+([\w\d_]+)\s+([\w\d_]+)'
        lines = netlist_str.strip().split('\n')
        
        # Eerst alle nodes registreren
        for line in lines:
            m = re.match(pattern, line.strip())
            if m:
                find(m.group(1)); find(m.group(2))
                # Als het een draad is: mergen
                if m.group(0).upper().startswith('W'):
                    union(m.group(1), m.group(2))
    
        # 3. Genereer de mapping tabel
        self.node_alias_map = {n: find(n) for n in parent}
    
        # 4. Genereer debug netlist string
        debug_lines = ["# DEBUG: Node Mapping Resolution"]
        for orig, master in self.node_alias_map.items():
            if orig != master:
                debug_lines.append(f"# {orig} -> {master}")
        
        debug_lines.append("# Normalized Netlist:")
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): continue
            match = re.match(pattern, line)
            if match:
                name, n1, n2 = match.groups()
                sn1, sn2 = find(n1), find(n2)
                rest = line[match.end():]
                status = "[MERGED]" if sn1 == sn2 and name.upper().startswith('W') else ""
                debug_lines.append(f"{name} {sn1} {sn2}{rest} {status}")
    
        self.debug_resolved_netlist = "\n".join(debug_lines)
        return self.node_alias_map


    def process_netlist(self, netlist_str):
        sim_lines = []
        draw_lines = []
        
        # Databases voor de PowerCircuit klasse
        switches = {} # { 'SW1': 'R_SW1' }
        diodes = {}   # { 'D1': {'v': 'V_D1', 'r': 'R_D1', 'int': 'int_D1'} }
        
        # Regex voor componenten en metadata
        #pattern = r'^(\w+)\s+(\w+)\s+(\w+)(?:\s+([^;]+))?(?:\s*;\s*(.*))?$'
        pattern = r'^(\w+)\s+([\w\d_]+)\s+([\w\d_]+)\s*([^;]*?)\s*(?:\s*;\s*(.*))?$'
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
                sim_lines.append(f"{v_sym} {n1} {int_node} {self.Vf} {l_str}")
                sim_lines.append(f"{r_sym} {int_node} {n2} {self.Rdr} {l_str}")
                
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
                sim_lines.append(f"{r_esr} {int_node} {n2} {esr} {l_str}")
                
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
        sim_netlist = self.clean_sim_netlist(os.linesep.join(sim_lines))
        return sim_netlist, os.linesep.join(draw_lines), switches, diodes


    def clean_sim_netlist(self, netlist_str, r_wire=1e-9):
        lines = [l.strip() for l in netlist_str.strip().split('\n') if l.strip() and not l.startswith('#')]
        
        parent = {}
        def find(n):
            parent.setdefault(n, n)
            if parent[n] == n: return n
            parent[n] = find(parent[n])
            return parent[n]
    
        def union(n1, n2):
            r1, r2 = find(n1), find(n2)
            if r1 != r2:
                # Priority: Merge underscore nodes into clean nodes
                if '_' in r1 and '_' not in r2: parent[r1] = r2
                else: parent[r2] = r1
    
        # Pass 1: Identify wires that should be collapsed (contain an underscore)
        components = []
        for line in lines:
            match = re.match(r'^(\w+)\s+(\w+)\s+(\w+)(.*)$', line)
            if not match: continue
            name, n1, n2, rest = match.groups()
            
            # Wires with underscores are "structural" and get collapsed
            if name.upper().startswith('W') and ('_' in n1 or '_' in n2):
                union(n1, n2)
            else:
                components.append((name, n1, n2, rest))
    
        # Pass 2: Rebuild netlist with resolved nodes
        sim_lines = []
        for name, n1, n2, rest in components:
            root1, root2 = find(n1), find(n2)
            
            if name.upper().startswith('W'):
                # Wire between two CLEAN nodes: convert to tiny R to keep nodes distinct
                sim_lines.append(f"R_WIRE_{name} {root1} {root2} {r_wire} {rest}")
            elif root1 != root2:
                # Standard component with mapped nodes
                sim_lines.append(f"{name} {root1} {root2} {rest}")
                
        return os.linesep.join(sim_lines)


    def draw(self, *args, **kwargs):
        from lcapy import Circuit
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

        self.Mshape = self.M0.shape
        self.Gshape = self.G0.shape
        self.Bshape = self.B0.shape

        states = (num_switch_states, num_diode_states)

        self.M = np.zeros(states + self.Mshape, dtype = float)
        self.G = np.zeros(states + self.Gshape, dtype = float)
        self.B = np.zeros(states + self.Bshape, dtype = float)
        
        self.Mdt = np.zeros(states + self.Mshape, dtype = float)
        self.Mmdt = np.zeros(states + self.Mshape, dtype = float)
        
        self.mgb_cache = np.zeros(states, dtype = bool)
        self.flush_be_cache(0)


    def mgb(self, sw_addr, d_addr, sw_array, d_array):
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
        #args = [RDB[s] for s in self.symbols]
        print(RDB)
        M, G, B = self.circuit.get_matrices(RDB)
        self.M[sw_addr, d_addr] = np.copy(M)
        self.G[sw_addr,d_addr] = np.copy(G)
        self.B[sw_addr, d_addr] = np.copy(B)
        self.mgb_cache[sw_addr, d_addr] = True


    def be(self, sw_addr, d_addr, sw_array, d_array, m, dt):
        if not self.mgb_cache[sw_addr][d_addr]:
            self.mgb(sw_addr, d_addr, sw_array, d_array)
        if m != self.m_cache:
            raise ValueError("cache was set up fot different m value " 
                             f"({self.m_cache}), flush it first")
        dt_micro = dt / m  # dt for microstepping
        
        Mdt = self.M[sw_addr][d_addr] / dt
        Mmdt = Mdt / m
        self.Mdt[sw_addr][d_addr] = Mdt
        self.Mmdt[sw_addr][d_addr] = Mmdt
        self.be_cache[sw_addr][d_addr] = True


    def flush_be_cache(self, m):
           states = (2 ** self.num_switches, 2 ** self.num_diodes)
           self.be_cache = np.zeros(states, dtype = bool)
           self.dt_cache = 0
           self.m_cache = m


    @property
    def ydict(self):
        d = {name : i for i, name in enumerate(self.expanded_outputs)}
        print("d", d)
        for d_name in self.diode_db:
            d_info = self.diode_db[d_name]
            del d["v_" + d_info["node"]]
            d["i_"+d_name] = d["i_" + d_info["V"]]
            del d["i_" + d_info["V"]]
            #del d["i_" + d_info["R"]]
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


    def expand_output(self, x, x_prev, dt):
        extra_i = []
        for c in self.sim_netlist.components:
            if c['type'] in ('R', 'C'):
                n1, n2 = c['n1'] - 1, c['n2'] - 1
                v1 = x[n1] if n1 >= 0 else 0
                v2 = x[n2] if n2 >= 0 else 0
                v_diff = v1 - v2
                
                if c['type'] == 'R':
                    # Wet van Ohm: i = G * v
                    r_val = c['val'] # rdb.get(c['name'], c['val'])
                    extra_i.append(v_diff / r_val)
                
                elif c['type'] == 'C':
                    v1_old = x_prev[n1] if n1 >= 0 else 0
                    v2_old = x_prev[n2] if n2 >= 0 else 0
                    v_old = v1_old - v2_old
                    
                    i_c = c['val'] * (v_diff - v_old) / dt
                    extra_i.append(i_c)
        return np.concatenate([x, extra_i])
                

    def sim_step(self, sw_addr, d_addr, sw_array, d_array, u_exp, x, m, t, dt, lc, result):
        x_prev = x
        d_array_prev = d_array
        
        # Try a dt step
        u = u_exp[:, [t]]
        if not self.be_cache[sw_addr][d_addr]:
            self.be(sw_addr, d_addr, sw_array, d_array, m, dt) # Loads into self.Ab, self.Bb
        Mdt = self.Mdt[sw_addr][d_addr]
        G = self.G[sw_addr][d_addr]
        B = self.B[sw_addr][d_addr]

        # Trial integration
        LHS = Mdt + G
        RHS = Mdt @ x + B @ u
        x_trial = np.linalg.solve(LHS, RHS)
        y_trial = self.expand_output(x, x_prev, dt)
                
        I_diodes = x_trial[self.diode_I_V_output_indices].ravel()
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
            dt_micro = dt / m
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
                    self.be(sw_addr, d_addr, sw_array, d_array, m, dt) # Loads into self.Ab, self.Bb
                Mmdt = self.Mmdt[sw_addr][d_addr]
                G = self.G[sw_addr][d_addr]
                B = self.B[sw_addr][d_addr]
                
                #Integrate microstep
                # Trial integration
                LHS = Mmdt + G
                RHS = Mmdt @ x + B @ u
                x = np.linalg.solve(LHS, RHS)
                #y = self.expand_output(x, x_prev, dt)
                
                # Check commutation within microstep
                I_diodes = x[self.diode_I_V_output_indices].T[0]
                d_array = (I_diodes > 0) * 1
                
                if (d_array != d_array_prev).any():
                    # Update topology mid-micro-loop
                    #print("commutation", d_array, t_micro[i] )
                    d_addr = int(np.dot(d_array, self.diode_addr))
                    d_array_prev = d_array
                    # Re-run this microstep with new topology
                    if not self.be_cache[sw_addr][d_addr]:
                        self.be(sw_addr, d_addr, sw_array, d_array, m, dt) 
                    Mmdt = self.Mmdt[sw_addr][d_addr]
                    G = self.G[sw_addr][d_addr]
                    B = self.B[sw_addr][d_addr]
                    LHS = Mmdt + G
                    RHS = Mmdt @ x + B @ u
                    x = np.linalg.solve(LHS, RHS)
            y = self.expand_output(x, x_prev, dt)
                
        result[:, [t]] = y
        #result[:self.num_sim_outputs, [t]] = x
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
            u_exp = np.ones((self.num_sim_inputs, n)) * self.Vf
            u_exp[self.sim_input_indices] = u
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
            if x.shape != (len(self.circuit.matrix_labels), 1):
                raise ValueError(f"x should have shape {self.Ashape}, "
                                 f"(got {x.shape}.")
        else:
            x = np.zeros((len(self.circuit.matrix_labels), 1))
            

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
V1 p m ; down
W1 p 1; right, size=2
W2 m 3; right , size=2
D0 0_0 1 ; rotate=45, size=1.5
D1 1 2 ; rotate=-45, size=1.5
D2 0_0 3 ; rotate=-45, size=1.5
D3 3 2 ; rotate=45, size=1.5
W3 2 4; right
W4 0_0 0_1 ; down, size=1.5
W5 0_1 0; right
C1 4 0 1000e-6 .1; down, size=1.5
R1 5 0_2 100; down
W7 4 5 ; right, size=1.5
W8 0 0_2; right
gnd 0 0_g ; down

"""

netlist_d = """
V1 1 0 3
D1 1 2
R1 2 0 1000
C2 2 0 1000e-6
#SW1 2 0
"""

netlist_d2 = """
Vin 1 0_1 0; down
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


pc = PowerCircuit(netlist_d2)

print("exp inp:")
#print(pc.expanded_inputs)
#print(pc.num_inputs)
print(pc.expanded_netlist)
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

y = pc.sim(u, [], n, dt, 1)


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
axf.plot(t, y["v_1"] - 0*y["v_2"], linewidth=1, label="Vin")
ax2.plot(t, y["i_R1"], linewidth=1, label="Idiode", color="tab:green")
axf.plot(t, y['v_2'], linewidth=1, label="Vout", color="tab:red")
axf.legend(loc="lower left")
ax2.legend()

img = mpimg.imread('circuit.png')
axs.imshow(img)
axs.axis('off')

plt.tight_layout()

plt.show()



