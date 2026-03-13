#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:24:34 2026

@author: Marcel Hesselberth
"""

import re
import numpy as np


class ExpressionParser:
    def __init__(self, variables=None):
        self.variables = variables or {}
        # Precedence: (priority, function)
        self.ops = {
            '+':   (1, np.add),
            '-':   (1, np.subtract),
            '*':   (2, np.multiply),
            '/':   (2, np.divide),
            '**':  (3, np.power),
            'u-':  (4, lambda x: -x)  # Unary minus has highest priority
        }
        self.funcs = {
            'sin': (1, np.sin), 'cos': (1, np.cos), 'exp': (1, np.exp),
            'abs': (1, np.abs), 'sqrt': (1, np.sqrt), 'sign': (1, np.sign),
            'ramp': (1, lambda t: np.maximum(0, t)),
            'pwm': (3, lambda t, d, f: (((t + 1e-15) % (1.0/f)) < (d/f)).astype(float)),
            'pulse': (4, lambda t, start, width, period:
                (((t - start + 1e-15) % period) < width).astype(float) * (t >= start).astype(float)),
            'where': (3, lambda cond, x, y: np.where(cond > 0, x, y))
        }

    def tokenize(self, expr):
        # Match numbers (also scientific), variables/functions, operators and ()
        return re.findall(r'[a-zA-Z_]\w*|[\d.]+(?:e[+-]?\d+)?|\*\*|[+\-*/(),]', expr)

    def to_postfix(self, expr_str):
        tokens = self.tokenize(expr_str)
        output = []
        op_stack = []
        last_was_operand = False # For difference between binary and unary -

        for tok in tokens:
            if re.match(r'^[\d.]+(?:e[+-]?\d+)?$', tok):
                output.append(('num', float(tok)))
                last_was_operand = True

            elif tok in self.funcs:
                op_stack.append(('func', tok))
                last_was_operand = False

            elif tok == '(':
                op_stack.append(('paren', '('))
                last_was_operand = False

            elif tok == ',':
                while op_stack and op_stack[-1] != ('paren', '('):
                    output.append(op_stack.pop())
                last_was_operand = False

            elif tok == ')':
                while op_stack and op_stack[-1] != ('paren', '('):
                    output.append(op_stack.pop())
                if not op_stack: raise ValueError("Mismatched parentheses")
                op_stack.pop() # Remove '('
                if op_stack and op_stack[-1][0] == 'func':
                    output.append(op_stack.pop())
                last_was_operand = True

            elif tok in '+-':
                if not last_was_operand:
                    if tok == '-': # Unaire minus
                        op_stack.append(('op', 'u-'))
                    # Ignore unary + 
                else: # Binary plus/minus
                    prec = self.ops[tok][0]
                    while op_stack and op_stack[-1][0] == 'op' and self.ops[op_stack[-1][1]][0] >= prec:
                        output.append(op_stack.pop())
                    op_stack.append(('op', tok))
                last_was_operand = False

            elif tok in self.ops: # *, /, **
                prec = self.ops[tok][0]
                while op_stack and op_stack[-1][0] == 'op':
                    top_prec = self.ops[op_stack[-1][1]][0]
                    # Exponentiation is right-associative
                    if (tok != '**' and top_prec >= prec) or (tok == '**' and top_prec > prec):
                        output.append(op_stack.pop())
                    else: break
                op_stack.append(('op', tok))
                last_was_operand = False

            else: # Variables
                output.append(('var', tok))
                last_was_operand = True

        while op_stack:
            item = op_stack.pop()
            if item[1] in '()': raise ValueError("Mismatched parentheses")
            output.append(item)

        return output


    def evaluate(self, postfix, t_array):
        if not postfix: 
            return np.zeros_like(t_array)
        
        stack = []
        ctx = {**self.variables, 't': t_array, 'pi': np.pi, 'e': np.e}
        
        for typ, val in postfix:
            try:
                if typ == 'num':
                    stack.append(np.full_like(t_array, val, dtype=float))
                elif typ == 'var':
                    if val not in ctx: 
                        raise ValueError(f"Unknown variable: '{val}'")
                    v = ctx[val]
                    # Convert variables to arrays
                    if not isinstance(v, np.ndarray):
                        v = np.full_like(t_array, float(v))
                    stack.append(v)
                elif typ == 'op':
                    if val == 'u-':
                        if len(stack) < 1: raise ValueError("Missing value for '-'")
                        stack.append(self.ops['u-'][1](stack.pop()))
                    else:
                        if len(stack) < 2: raise ValueError(f"Not enough values for '{val}'")
                        b, a = stack.pop(), stack.pop()
                        stack.append(self.ops[val][1](a, b))
                elif typ == 'func':
                    n_args, func = self.funcs[val]
                    if len(stack) < n_args:
                        raise ValueError(f"Missing argument for '{val}' (expected {n_args})")
                    args = [stack.pop() for _ in range(n_args)][::-1]
                    stack.append(func(*args))
            except Exception as e:
                # Also catches numpy errors
                raise ValueError(f"{e}")

        if len(stack) != 1:
            raise ValueError(f"Invalid expression syntax (stack size: {len(stack)})")
            
        return stack[0] 




# Example usage:
if __name__ == "__main__":
    t = np.linspace(0, 1, 100)
    parser = ExpressionParser({'Vcc': 5, 'tau': 0.1, 'f': 50, 'phase': 0, 'omega': 100, 'Vdd': 12})
    
    test_expr = "Vcc * (1 - exp(-t / tau))"
    postfix = parser.to_postfix(test_expr)
    result = parser.evaluate(postfix, t)
    print(f"Resultaat van '{test_expr}' (eerste 5 waarden):", result[:5])


import re
import numpy as np

class ExpressionParser:
    def __init__(self, variables=None):
        self.variables = variables or {}
        # Precedence: (priority, function)
        self.ops = {
            '+':   (1, np.add),
            '-':   (1, np.subtract),
            '*':   (2, np.multiply),
            '/':   (2, np.divide),
            '**':  (3, np.power),
            'u-':  (4, lambda x: -x)
        }
        self.funcs = {
            'sin': (1, np.sin), 'cos': (1, np.cos), 'exp': (1, np.exp),
            'abs': (1, np.abs), 'sqrt': (1, np.sqrt), 'sign': (1, np.sign),
            'ramp': (1, lambda t: np.maximum(0, t)),
            'pwm': (3, lambda t, d, f: (((t + 1e-12) % (1.0/f)) < (d/f)).astype(float)),
            'pulse': (4, lambda t, start, width, period:
                (((t - start + 1e-12) % period) < width).astype(float) * (t >= start).astype(float)),
            'where': (3, lambda cond, x, y: np.where(cond > 0, x, y))
        }

    def tokenize(self, expr):
        if not expr or not expr.strip():
            raise ValueError("Empty expression")
            
        # Updated regex: scientific notation must be followed by digits
        pattern = r'[\d.]+(?:[eE][+-]?\d+)|[a-zA-Z_]\w*|[\d.]+|\*\*|[+\-*/(),]'
        tokens = re.findall(pattern, expr)
        
        # Verify that we matched the entire non-whitespace content
        if len("".join(tokens)) != len(re.sub(r'\s+', '', expr)):
            raise ValueError("Invalid characters or malformed tokens")
            
        return tokens

    def to_postfix(self, expr_str):
        tokens = self.tokenize(expr_str)
        output = []
        op_stack = []
        last_was_operand = False 

        for tok in tokens:
            is_num = re.match(r'^[\d.]+(?:[eE][+-]?\d+)?$', tok)
            is_var_func = tok in self.funcs or (re.match(r'^[a-zA-Z_]\w*$', tok) and tok not in self.ops)

            # State check: No two operands in a row (e.g., "5 5", "5 sin")
            if (is_num or is_var_func or tok == '('):
                if last_was_operand and tok != '(':
                    raise ValueError(f"Missing operator before '{tok}'")

            # State check: Binary operator must have a left operand
            if tok in self.ops and tok not in '+-' and not last_was_operand:
                raise ValueError(f"Operator '{tok}' missing left operand")

            if is_num:
                output.append(('num', float(tok)))
                last_was_operand = True

            elif tok in self.funcs:
                op_stack.append(('func', tok))
                last_was_operand = False

            elif tok == '(':
                op_stack.append(('paren', '('))
                last_was_operand = False

            elif tok == ',':
                while op_stack and op_stack[-1] != ('paren', '('):
                    output.append(op_stack.pop())
                if not op_stack: raise ValueError("Comma outside of function arguments")
                last_was_operand = False

            elif tok == ')':
                while op_stack and op_stack[-1] != ('paren', '('):
                    output.append(op_stack.pop())
                if not op_stack: raise ValueError("Mismatched parentheses")
                op_stack.pop() 
                if op_stack and op_stack[-1][0] == 'func':
                    output.append(op_stack.pop())
                last_was_operand = True

            elif tok in '+-':
                if not last_was_operand:
                    if tok == '-': 
                        op_stack.append(('op', 'u-'))
                else: 
                    prec = self.ops[tok][0]
                    while op_stack and op_stack[-1][0] == 'op' and self.ops[op_stack[-1][1]][0] >= prec:
                        output.append(op_stack.pop())
                    op_stack.append(('op', tok))
                last_was_operand = False

            elif tok in self.ops: 
                prec = self.ops[tok][0]
                while op_stack and op_stack[-1][0] == 'op':
                    top_prec = self.ops[op_stack[-1][1]][0]
                    if (tok != '**' and top_prec >= prec) or (tok == '**' and top_prec > prec):
                        output.append(op_stack.pop())
                    else: break
                op_stack.append(('op', tok))
                last_was_operand = False

            else: # Variable
                output.append(('var', tok))
                last_was_operand = True

        while op_stack:
            item = op_stack.pop()
            if item[0] == 'paren': raise ValueError("Mismatched parentheses")
            output.append(item)
            
        if not last_was_operand:
            raise ValueError("Expression ends prematurely or is incomplete")

        return output

    def evaluate(self, postfix, t_array):
        if not postfix: raise ValueError("Empty expression")
        stack = []
        ctx = {**self.variables, 't': t_array, 'pi': np.pi, 'e': np.e}
        
        for typ, val in postfix:
            if typ == 'num':
                stack.append(np.full_like(t_array, val, dtype=float))
            elif typ == 'var':
                if val not in ctx: raise ValueError(f"Unknown variable: '{val}'")
                v = ctx[val]
                if not isinstance(v, np.ndarray):
                    v = np.full_like(t_array, float(v))
                stack.append(v)
            elif typ == 'op':
                if val == 'u-':
                    if len(stack) < 1: raise ValueError("Missing value for '-'")
                    stack.append(self.ops['u-'][1](stack.pop()))
                else:
                    if len(stack) < 2: raise ValueError(f"Not enough values for '{val}'")
                    b, a = stack.pop(), stack.pop()
                    stack.append(self.ops[val][1](a, b))
            elif typ == 'func':
                n_args, func = self.funcs[val]
                if len(stack) < n_args:
                    raise ValueError(f"Missing argument for '{val}'")
                args = [stack.pop() for _ in range(n_args)][::-1]
                stack.append(func(*args))

        if len(stack) != 1:
            raise ValueError(f"Invalid expression syntax (stack size: {len(stack)})")
        return stack[0]

if __name__ == "__main__":
    tests = [
        "2 + 3 * 4",
        "-5 + +7",
        "+3.14 * -2",
        "2 ** 3 ** 2",
        "4 / 2 ** 3",
        "- - -2",
        "- -2 ** 3",
        "+ - + - 5",
        "sin(pi/2)",
        "cos(0)",
        "exp(1)",
        "sqrt(16)",
        "ramp(-2)",
        "ramp(t - 1)",
        "pwm(t, 0.5, 1000)",
        "pulse(t, 0, 0.2, 1)",
        "pwm(t, 0.3, 500)",
        "pulse(t + 1, -0.5, 0.1, 2)",
        "Vcc * (1 - exp(-t / tau))",
        "2 * sin(2 * pi * f * t + phase)",
        "Vdd / 2 + 1.2 * cos(omega * t)",
        "(2 + 3) * sin(pi * t) + pwm(t, 0.4, 200)",
        "pulse(t, 0, 0.1, 1) * ramp(t - 0.5)",
        "sign(-t + 1) * sqrt(9 + t**2)",
        "exp(-abs(t)) * cos(2*pi*50*t)",
        "(1+2)*3",
        "1+(2*3)"
    ]

    variables = {
        'Vcc':   5.0,
        'Vdd':   3.3,
        'tau':   0.001,
        'f':     1000,
        'freq':  50,
        'omega': 2 * np.pi * 50,
        'phase': np.pi / 4,
    }

    parser = ExpressionParser(variables=variables)
    t = np.linspace(0, 0.01, 5)

    print("Expression parser test suite")
    print("t =", t.round(4))
    print("-" * 70)

    for expr in tests:
        try:
            postfix = parser.to_postfix(expr)
            result = parser.evaluate(postfix, t)

            if np.allclose(result, result[0], rtol=1e-5, atol=1e-8):
                val_str = f"{result[0]:.6g}"
            else:
                val_str = ", ".join(f"{x:.4g}" for x in result) + " …"

            print(f"{expr:48}  →  {val_str}")
        except Exception as e:
            print(f"{expr:48}  →  ERROR: {str(e)}")
    
    # Definieer de variabelen voor de tests
    params = {
        'Vcc': 5.0, 'tau': 0.01, 'f': 50.0, 'phase': 0.0, 
        'omega': 100.0, 'Vdd': 12.0
    }
    t = np.array([0.0, 0.0025, 0.005, 0.0075, 0.01])
    parser = ExpressionParser(params)
    
    # Lijst met foute inputs om op robuustheid te testen
    bad_tests = [
        "3 + * 4",                 # Dubbele operator / ontbrekende operand
        "sin(1, 2)",               # Te veel argumenten voor sin
        "pwm(t, 0.5)",             # Te weinig argumenten voor pwm (verwacht 3)
        "5 + onbekende_var",       # Niet-bestaande variabele
        "(2 + 3",                  # Mismatched parentheses (openend)
        "2 + 3)",                  # Mismatched parentheses (sluitend)
        "4 / (2 - 2)",             # Delen door nul (Numpy geeft meestal 'inf' of 'nan', geen crash)
        "abs()",                   # Lege functie-aanroep
        "5 5",                     # Twee getallen zonder operator (stack houdt > 1 item over)
        ", 2 + 3",                 # Losse komma aan het begin
        "sqrt(-1)",                # Negatieve wortel (Numpy geeft 'nan')
        "+ +"
    ]
    
    print(f"{'Expressie':<30} | {'Status':<10} | {'Resultaat / Foutmelding'}")
    print("-" * 80)
    
    for expr in bad_tests:
        try:
            postfix = parser.to_postfix(expr)
            result = parser.evaluate(postfix, t)
            # Als het hier komt, is het technisch 'geslaagd' (bijv. bij nan/inf)
            print(f"{expr:<30} | {'OK':<10} | {result}")
        except Exception as e:
            # Hier vangen we de fouten op die we zelf in de parser hebben ingebouwd
            print(f"{expr:<30} | {'FOUT':<10} | {e}")

