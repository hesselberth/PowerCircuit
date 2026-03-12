#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 23:54:43 2026

@author: Marcel Hesselberth
"""

import re
import math
import numpy as np
import tempfile
import os
from lcapy import Circuit
import matplotlib.image as mpimg

class ExpressionParser:
    def __init__(self, variables=None):
        self.variables = variables or {}
        self.ops = {
            '+': (1, np.add), '-': (1, np.subtract), '*': (2, np.multiply),
            '/': (2, np.divide), '**': (3, np.power)
        }
        self.funcs = {
            'sin': (1, np.sin), 'cos': (1, np.cos), 'exp': (1, np.exp),
            'sqrt': (1, np.sqrt), 'sign': (1, np.sign),
            'pwm': (3, lambda t, d, f: (((t + 1e-15) % (1.0/f)) < (d/f)).astype(float)),
            'ramp': (1, lambda t: np.maximum(0, t)),
            'pulse': (4, lambda t, start, width, period:
                (((t - start + 1e-15) % period) < width).astype(float) * (t >= start).astype(float))
        }

    def tokenize(self, expr):
        return re.findall(r'[a-zA-Z_]\w*|[\d.]+(?:e[+-]?\d+)?|\*\*|[+\-*/(),]', expr)

    def to_postfix(self, expr_str):
        tokens = self.tokenize(expr_str)
        output, stack = [], []
        for token in tokens:
            if re.match(r'^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$', token):
                output.append(('num', float(token)))
            elif token in self.funcs:
                stack.append(token)
            elif token == '(':
                stack.append('(')
            elif token == ',':
                while stack and stack[-1] != '(':
                    top = stack.pop()
                    output.append(('op' if top in self.ops else 'func', top))
            elif token == ')':
                while stack and stack[-1] != '(':
                    top = stack.pop()
                    output.append(('op' if top in self.ops else 'func', top))
                if stack:
                    stack.pop()
                if stack and stack[-1] in self.funcs:
                    output.append(('func', stack.pop()))
            elif token in self.ops:
                prec, _ = self.ops[token]
                while stack and stack[-1] in self.ops and self.ops[stack[-1]][0] >= prec:
                    output.append(('op', stack.pop()))
                stack.append(token)
            else:
                output.append(('var', token))
        while stack:
            top = stack.pop()
            output.append(('op' if top in self.ops else 'func', top))
        return output

    def evaluate(self, postfix, t_array):
        if not postfix:
            return np.zeros_like(t_array)
        stack = []
        context = {**self.variables, 't': t_array, 'pi': np.pi, 'e': np.e}
        for p_type, val in postfix:
            if p_type == 'num':
                stack.append(np.full_like(t_array, val))
            elif p_type == 'var':
                v = context.get(val, 0.0)
                if not isinstance(v, np.ndarray):
                    v = np.full_like(t_array, float(v))
                stack.append(v)
            elif p_type == 'op':
                if len(stack) < 2:
                    raise ValueError(f"Missing operands for {val}")
                b, a = stack.pop(), stack.pop()
                func = self.ops[val][1]
                stack.append(func(a, b))
            elif p_type == 'func':
                n_args, func = self.funcs[val]
                if len(stack) < n_args:
                    raise ValueError(f"Missing arguments for {val}")
                args = [stack.pop() for _ in range(n_args)][::-1]
                stack.append(func(*args))
        return stack[0] if stack else np.zeros_like(t_array)


class Netlist:
    """
    A PowerCircuit compatible netlist.

    Format: <component> <+node> <-node>
    Example: R1 node1 node2

    Nodes may be numbers but they are treated as strings. Netlist will perform
    the contiguous node number assignment for matrix processing.
    Component declarations can contain meta information (like drawing hints):
    V1 3 0; down, size = 1.5

    And comments:
    D1 4 5; rotate=45 # Part of bridge rectifier 1
    SW1 Vin Vout 0 # Off by default
    
    Capacitors and inductors can have an ESR specified:
    C1 1 2 1u 0.1  # this cap has a 0.1 ohm ESR
    L1 1 2 1u 100m  # and this inductor too. SI prefixes are supported.
    
    Variable inputs may be specified in the simulation command or in the
    netlist:
    Iload 5 0 1 + sin(t)
    
    Besides component declarations the netlist supports variable declarations:
    f = 1000
    w = 2 * pi * f # pi is a predefined number, so is e
    ripple = 1
    dc = 12
    
    Vin 1 0 dc + ripple * cos(w*t)
    
    Meta-only lines are allowed
    ; style=american
    
    And finally GND is shorthand for a ground symbol (node 0)
    gnd 0 0_g; down
    """
    prefixes = {12: 'T', 9: 'G', 6: 'M', 3: 'k', 0: '', -3: 'm', -6: 'u', -9: 'n', -12: 'p', -15: 'f'}

    def __init__(self, netlist_str):
        self.components = []
        self.variables = {}
        self.node_names = ['0']
        self.node_map = {'0': 0}
        self.standalone_meta = []

        self.parse(netlist_str)
        self._node_rename_map = self.get_node_rename_map()  # cache

    def _get_node_id(self, name):
        if name not in self.node_map:
            self.node_map[name] = len(self.node_names)
            self.node_names.append(name)
        return self.node_map[name]

    def _pretty_prefix(self, val):
        if not isinstance(val, (int, float)) or val == 0:
            return str(val)
        abs_val = abs(val)
        exponent = math.floor(math.log10(abs_val))
        eng_exponent = int((exponent // 3) * 3)
        prefix = self.prefixes.get(eng_exponent, f"e{eng_exponent}")
        scaled = val / (10 ** eng_exponent)
        return f"{scaled:g}{prefix}"

    def _interpret_value(self, val_str):
        val_str = val_str.strip()
        if not val_str:
            return 0.0
        suffix_map = {v: 10**k for k, v in self.prefixes.items() if v}
        if len(val_str) > 1 and val_str[-1] in suffix_map:
            try:
                return float(val_str[:-1]) * suffix_map[val_str[-1]]
            except:
                pass
        try:
            return float(val_str)
        except:
            return val_str

    def parse(self, netlist_str):
        line_num = 1
        for line in netlist_str.strip().split('\n'):
            line = line.split('#')[0].strip()  # remove comments
            if not line:
                continue
            dm = line.split(';', 1)  # line = data ; optional_meta
            data, meta = dm[0].strip(), (dm[1].strip() if len(dm) > 1 else "")

            if not data and meta:  # directive
                self.standalone_meta.append(meta)
                continue

            # variables first to avoid collision with components
            if '=' in data and ' ' not in data.split('=')[0].strip():
                var_name, val_str = data.split('=', 1)
                self.variables[var_name.strip()] = self._interpret_value(val_str.strip())
                continue

            parts = re.split(r'\s+', data)
            if len(parts) < 3:
                raise ValueError(f"Syntax error in line {line_num}:\n{line}")

            name = parts[0]
            n1 = self._get_node_id(parts[1])
            n2 = self._get_node_id(parts[2])
            rem = parts[3:]

            if name.lower() == 'gnd':
                name = "W_gndsink"
                n1 = self._get_node_id(parts[1] if len(parts) > 1 else "0")
                n2 = self._get_node_id("0_g")
                val = 0.0
                esr = 0.0
                meta = f"ground, size=0.0, {meta}".strip(", ")
            else:
                if name.upper().startswith(('C', 'L')) and len(rem) >= 2:
                    val = self._interpret_value(" ".join(rem[:-1]))
                    esr = self._interpret_value(rem[-1])
                else:
                    val = self._interpret_value(" ".join(rem))
                    esr = 0.0

            self.components.append({
                'name': name,
                'type': name[0].upper(),
                'n1': n1,
                'n2': n2,
                'val': val,
                'esr': esr,
                'meta': meta
            })
            line_num += 1

    def get_node_rename_map(self):
        adj = {name: {name} for name in self.node_names}
        for c in self.components:
            if c['type'] == 'W' or "ground" in c.get('meta', ''):
                n1_n = self.node_names[c['n1']]
                n2_n = self.node_names[c['n2']]
                u = adj[n1_n] | adj[n2_n]
                for node in u:
                    adj[node] = u
        return {n: ('0' if '0' in adj[n] else min(adj[n])) for n in self.node_names}

    def _should_skip_in_sim(self, c):
        return c['type'] == 'W' or "ground" in c.get('meta', '')

    def _should_skip_value_in_draw(self, c):
        ctype = c['name'][0].upper()
        val = c['val']
        if self._should_skip_in_sim(c):
            return True
        if isinstance(val, str) and ctype in ('V', 'I'):
            return True
        if isinstance(val, (int, float)) and val == 0:
            return True
        return False

    def _get_value_string_draw(self, c):
        if self._should_skip_value_in_draw(c):
            return ""
        return self._pretty_prefix(c['val'])

    def _get_lcapy_annotations(self, c):
        parts = []
        name = c['name']
        ctype = name[0].upper()
        val = c['val']
        pretty_name = f"{name[0]}_{{{name[1:]}}}" if len(name) > 1 else name

        if isinstance(val, str) and ctype in ('V', 'I'):
            parts.append(f"l={{{pretty_name}(t)}}")
        elif ctype == 'R':
            parts.append(f"l={{{pretty_name}={self._pretty_prefix(val)}}}")
        elif ctype == 'D':
            parts.append(f"l={{{pretty_name}}}")
        elif ctype in ('C', 'L') and c.get('esr'):
            v_str = self._pretty_prefix(val)
            e_str = self._pretty_prefix(c['esr'])
            parts.append(f"l={{{pretty_name}={v_str}}}")
            parts.append(f"a={{{e_str}\\Omega}}")
        return parts

    def get_draw_netlist(self):
        rmap = self._node_rename_map
        lines = [f"# {k}={self._pretty_prefix(v)}" for k, v in self.variables.items()]

        for c in self.components:
            def gn(idx):
                n = self.node_names[idx]
                return n if n == rmap[n] or n.startswith('_') else f"_{n}"

            name = c['name']
            n1_str = gn(c['n1'])
            n2_str = gn(c['n2'])
            meta = c.get('meta', '')

            val_str = self._get_value_string_draw(c)
            annotations = self._get_lcapy_annotations(c)
            if meta:
                annotations.append(meta)

            line = f"{name} {n1_str} {n2_str}"
            if val_str:
                line += f" {val_str}"
            if annotations:
                line += f" ; {', '.join(annotations)}"

            lines.append(line)

        lines.extend([f"; {m}" for m in self.standalone_meta])
        return "\n".join(lines).lstrip()

    def _extract_layout(self, meta_str):
        layout_keywords = ('up', 'down', 'left', 'right', 'rotate', 'angle', 'size', 'at', 'color')
        if not meta_str:
            return ""
        parts = [p.strip() for p in meta_str.split(',')]
        layout = [p for p in parts if any(k in p for k in layout_keywords)]
        return f"; {', '.join(layout)}" if layout else ""

    def _get_sim_value_str(self, val):
        if isinstance(val, (int, float)):
            return self._pretty_prefix(val)
        return str(val)

    def _generate_specialized_lines(self, c, n1, n2, layout_str):
        lines = []
        name = c['name']
        if c['type'] == 'D':
            int_node = f"int_{name}"
            lines.append(f"V_{name} {n1} {int_node} 0.7 {layout_str}".strip())
            lines.append(f"R_{name} {int_node} {n2} 10m {layout_str}".strip())
        elif c['type'] in ('C', 'L') and c.get('esr'):
            int_node = f"int_{name}"
            val_str = self._pretty_prefix(c['val'])
            esr_str = self._pretty_prefix(c['esr'])
            lines.append(f"{name} {n1} {int_node} {val_str} {layout_str}".strip())
            lines.append(f"R_esr_{name} {int_node} {n2} {esr_str} {layout_str}".strip())
        elif name.upper().startswith('SW'):
            r_val = self._get_sim_value_str(c['val']) or f"{{R_{name}}}"
            lines.append(f"R_{name} {n1} {n2} {r_val} {layout_str}".strip())
        return lines

    def get_sim_netlist(self):
        rmap = self._node_rename_map
        lines = [f"{k}={self._pretty_prefix(v)}".strip() for k, v in self.variables.items()]

        for c in self.components:
            if self._should_skip_in_sim(c):
                continue

            n1 = rmap[self.node_names[c['n1']]]
            n2 = rmap[self.node_names[c['n2']]]
            layout_str = self._extract_layout(c.get('meta', ''))

            special_lines = self._generate_specialized_lines(c, n1, n2, layout_str)
            if special_lines:
                lines.extend(special_lines)
                continue

            val_str = self._get_sim_value_str(c['val'])
            line = f"{c['name']} {n1} {n2} {val_str} {layout_str}".strip()
            lines.append(line)

        return "\n".join(lines)

    def get_signal_func(self, name):
        for c in self.components:
            if c['name'] == name:
                parser = ExpressionParser(self.variables)
                postfix = parser.to_postfix(str(c['val']))
                return lambda t: parser.evaluate(postfix, t)
        raise KeyError(f"Component '{name}' not found")

    def draw_to_axis(self, ax):
        draw_netlist = self.get_draw_netlist()
        # Filter lines that lcapy might not understand
        clean_lines = [l for l in draw_netlist.split('\n') if '=' not in l.split(';')[0]]
        cct = Circuit('\n'.join(clean_lines))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_png = os.path.join(tmpdir, 'schematic.png')
            cct.draw(tmp_png, dpi=300)
            img = mpimg.imread(tmp_png)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Circuit Schematic")

    def __str__(self):
        return self.get_draw_netlist()


# ────────────────────────────────────────────────
# Example usage (unchanged)
# ────────────────────────────────────────────────

if __name__ == "__main__":
    netlist_fb = """
    V1 p m sin(10*t); down
    W1 p 1; right, size=2
    W2 m 3; right , size=2
    D0 0_0 1 ; rotate=45, size=1.5
    D1 1 2 ; rotate=-45, size=1.5
    D2 0_0 3 ; rotate=-45, size=1.5
    D3 3 2 ; rotate=45, size=1.5
    W3 2 4; right, size=1
    W4 0_0 0_1 ; down, size=1.5
    W5 0_1 0; right
    C1 4 0 1000e-6 .1; down, size=1.5
    R1 5 0_2 100; down
    W7 4 5 ; right, size=2
    W8 0 0_2; right
    gnd 0 0_g ; down
    """

    netlist_d2 = """
    Vin 1 0_1 10; down
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
    
    nl = Netlist(netlist_fb)
    print("--- Sim Netlist ---")
    print(nl.get_sim_netlist())
    print("\n--- Draw Netlist ---")
    print(nl.get_draw_netlist())
    
    nl = Netlist(netlist_d2)
    print("--- Sim Netlist ---\n")
    print(nl.get_sim_netlist())
    print("\n--- Draw Netlist ---\n")
    print(nl.get_draw_netlist())


    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 1]})
    
    axd = ax[0]
    
    nl.draw_to_axis(axd)
    
    axd.axis('off')
    
    plt.tight_layout()
    
    plt.show()
