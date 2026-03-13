#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:54:20 2026

@author: Marcel Hesselberth
"""

import pytest
import numpy as np
import sys, os
sys.path.append(os.getcwd() + '/..')
from parser import ExpressionParser


# ───────────────────────────────────────────────
# Fixtures & helpers
# ───────────────────────────────────────────────

@pytest.fixture
def variables():
    return {
        'Vcc': 5.0,
        'Vdd': 12.0,
        'tau': 0.01,
        'f': 50.0,
        'omega': 100.0,
        'phase': 0.0,
        'pi': np.pi,
        'e': np.e,
    }


@pytest.fixture
def parser(variables):
    return ExpressionParser(variables=variables)


@pytest.fixture
def t_small():
    return np.array([0.0, 0.002, 0.005, 0.01])


@pytest.fixture
def t_empty():
    return np.array([])


RTOL = 1e-9
ATOL = 1e-12


def assert_array_close(actual, expected):
    if np.isscalar(expected):
        expected = np.full_like(actual, expected)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


# ───────────────────────────────────────────────
# Positive tests – should succeed
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_at_0", [
    ("2 + 3", 5.0),
    ("-5 + 7.5", 2.5),
    ("2 * -3", -6.0),
    ("2 ** 3 ** 2", 512.0),
    ("- - - -8", 8.0),
    ("sin(pi/2)", 1.0),
    ("abs(-3.2)", 3.2),
    ("sqrt(16)", 4.0),
    ("ramp(-2)", 0.0),
    ("Vcc * (1 - exp(-t / tau))", 0.0),           # at t=0
    ("1.23e-4 + 5", 5.000123),
    ("-4.5e+2 * 2", -900.0),
    (".5e-1", 0.05),
    ("-sqrt(16)", -4.0),
    ("--abs(-5)", 5.0),
])
def test_correct_expressions(parser, t_small, expr, expected_at_0):
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_small)
    assert_array_close(result[0], expected_at_0)


# ───────────────────────────────────────────────
# Error cases 
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_exc_msg_substring", [
    # Mismatched parentheses
    ("(1 + 2",           "Mismatched parentheses"),
    ("1 + 2)",           "Mismatched parentheses"),
    ("((3 * 4)",         "Mismatched parentheses"),
    (") + 5",            "Mismatched parentheses"),

    # Wrong number of function arguments
    ("sin()",            "Missing argument for 'sin'"),
    ("sin(1,2)",         "Invalid expression syntax (stack size: 2)"),
    ("pwm(t)",           "Missing argument for 'pwm'"),
    ("pwm(t, 0.5)",      "Missing argument for 'pwm'"),
    ("pulse(t,0,0.2)",   "Missing argument for 'pulse'"),

    # Syntax / missing operands
    ("3 + * 4",          "Operator '*' missing left operand"),
    ("5 / / 2",          "Operator '/' missing left operand"),
    ("2 3 + 4",          "Missing operator before '3'"),
    ("+ +",              "Expression ends prematurely or is incomplete"),

    # Unknown variables
    ("x + 5",            "Unknown variable: 'x'"),
    ("Vcc + unknown",    "Unknown variable: 'unknown'"),

    # Comma outside function (tends to cause stack or paren error)
    (", 5",              "Comma outside of function arguments"),
    ("5 + , 3",          "Comma outside of function arguments"),
    ("(1, 2)",           "Invalid expression syntax (stack size: 2)"),

    # Malformed numbers — current tokenizer accepts many as variables → later fails
    ("1.2.3",            "could not convert string to float: '1.2.3'"),
    ("1e",               "Missing operator before 'e'"),
    ("1e+",              "Missing operator before 'e'"),
    ("e4",               "Unknown variable: 'e4'"),
])
def test_error_cases_real_messages(parser, t_small, expr, expected_exc_msg_substring):
    with pytest.raises(ValueError) as exc_info:
        postfix = parser.to_postfix(expr)
        _ = parser.evaluate(postfix, t_small)

    assert expected_exc_msg_substring in str(exc_info.value)


# ───────────────────────────────────────────────
# Numerical edge cases (should not raise — produce inf/nan)
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr", [
    "1 / 0",
    "sqrt(-1)",
    "1 / (t - t)",
    "exp(1000)",
])
def test_numerical_edges_no_exception(parser, t_small, expr):
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_small)
    assert len(result) == len(t_small)
    # We allow inf / nan
    assert np.all(np.isfinite(result) | np.isnan(result) | np.isinf(result))


# ───────────────────────────────────────────────
# Empty input array
# ───────────────────────────────────────────────

def test_empty_t_array(parser, t_empty):
    expr = "Vcc * sin(2 * pi * f * t + phase)"
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_empty)
    assert result.shape == (0,)
    assert result.dtype.kind == 'f'


def test_empty_expression(parser, t_small):
    with pytest.raises(ValueError):
        # Current code returns zeros_like → but many would prefer error
        # If you want to make it raise → change evaluate when postfix empty
        result = parser.evaluate([], t_small)
        assert np.all(result == 0)


# ───────────────────────────────────────────────
# Deep nesting (should usually work until stack/recursion limit)
# ───────────────────────────────────────────────

def test_deep_nesting(parser, t_small):
    # ~20 levels — should be fine
    expr = "sin(" * 10 + "t" + ")" * 10
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_small)
    assert len(result) == len(t_small)













