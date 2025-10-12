# Run this file by create the following bash file: bash run_euler.sh
# -------------------- run_euler.sh --------------------
# theta=1

# F='x*y - x**3'
# EXACT='x**2 + 2 - np.exp(x**2/2)'  

# python euler_cli.py --py "$F" -a 0 -b 2 -N 50 \
#   --method euler_forward --theta $theta \
#   --plot --save-plot euler_forward.png \
#   --exact "$EXACT"

# F='x*y - x**3'
# EXACT='x**2 + 2 - 5*np.exp(x**2/2 - 2)'
# python euler_cli.py --py "$F" -a 0 -b 2 -N 50 \
#   --method euler_backward --theta $theta \
#   --plot --save-plot euler_backward.png --exact "$EXACT"

# -------------------- euler_cli.py --------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve y'(x) = f(x, y) with two methods:
  - euler_forward  uses theta as y(a)=theta
  - euler_backward uses theta as y_{N-1}=theta at x_{N-1}=b-h

Supply f(x, y) by one of:
  --py "<expr in x,y>", or
  --py-func "<code that defines def f(x, y): ...>", or
  --py-file path.py    # file containing def f(x, y)

Optional exact solution overlay:
  --exact "<expr in x>"  vectorized with numpy is recommended

Examples
--------
F='x*y - x**3'
# forward: y(a)=theta
python euler_cli.py --py "$F" -a 0 -b 2 -N 50 \
  --method euler_forward --theta 1 --plot --save-plot fwd.png

# backward: y_{N-1}=theta at x=b-h
theta=1
python euler_cli.py --py "$F" -a 0 -b 2 -N 50 \
  --method euler_backward --theta $theta --plot --save-plot bwd.png \
  --exact "x**2 + 2 + ( $theta - (2**2 + 2) )*np.exp(x**2/2 - (2**2)/2)"
"""

from __future__ import annotations
from typing import Callable, Optional
import argparse
import csv
import sys
import math
import numpy as np


# ---------------- ODE solvers ----------------

def euler_forward(f: Callable[[float, float], float],
                  a: float, b: float, theta: float, N: int):
    if N <= 0:
        raise ValueError("N must be positive")
    h = (b - a) / float(N)
    X = np.linspace(a, b, N + 1, dtype=float)
    Y = np.zeros(N + 1, dtype=float)
    Y[0] = float(theta)  # y(a)=theta
    for i in range(N):
        Y[i + 1] = Y[i] + h * f(float(X[i]), float(Y[i]))
    return X, Y


def euler_backward(f: Callable[[float, float], float],
                   a: float, b: float, theta: float, N: int):
    """
    Backward stepping with boundary at the last interior node:
      grid: X_i = a + i*h, i=0..N-1 with h=(b-a)/N  (endpoint not included)
      set Y_{N-1} = theta, then for i=N-1..1:
         Y_{i-1} = Y_i - h * f(X_i, Y_i)
    Returns X of length N and Y of length N.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    h = (b - a) / float(N)
    # N points, endpoint excluded so last x is b-h
    X = np.linspace(a, b, N, endpoint=False, dtype=float)
    Y = np.zeros(N, dtype=float)
    Y[-1] = float(theta)  # y_{N-1} = theta at x = b - h
    for i in range(N - 1, 0, -1):
        xi = float(X[i])
        yi = float(Y[i])
        Y[i - 1] = yi - h * f(xi, yi)
    return X, Y


SOLVERS = {
    "euler_forward": euler_forward,
    "euler_backward": euler_backward,
}


# ---------------- Safe builders ----------------

def _safe_env():
    safe_builtins = {}
    env = {
        "__builtins__": safe_builtins,
        "math": math,
        "np": np,
        "numpy": np,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "exp": math.exp, "log": math.log, "log10": math.log10,
        "sqrt": math.sqrt, "fabs": math.fabs, "pow": pow,
        "pi": math.pi, "e": math.e,
    }
    return env


def build_f_from_py_expr(expr: str) -> Callable[[float, float], float]:
    code = f"lambda x, y: ({expr})"
    env = _safe_env()
    try:
        fn = eval(code, env, {})
    except Exception as e:
        raise ValueError(f"Cannot compile f expression: {e}")
    if not callable(fn):
        raise TypeError("Expr for f does not compile to a callable")
    return fn


def build_f_from_py_func(src: str) -> Callable[[float, float], float]:
    env = _safe_env()
    loc = {}
    try:
        exec(src, env, loc)
    except Exception as e:
        raise ValueError(f"Cannot exec f code: {e}")
    f = loc.get("f", env.get("f"))
    if not callable(f):
        raise ValueError("No function f(x, y) found after exec")
    return f


def build_f_from_file(path: str) -> Callable[[float, float], float]:
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    return build_f_from_py_func(src)


def build_y_exact_from_expr(expr: str):
    code = f"lambda x: ({expr})"
    env = _safe_env()
    try:
        fn = eval(code, env, {})
    except Exception as e:
        raise ValueError(f"Cannot compile exact expression: {e}")
    if not callable(fn):
        raise TypeError("Expr for exact does not compile to a callable")
    return fn


# ---------------- CLI ----------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Solve y'(x)=f(x,y) with Euler forward or backward. Exact overlay optional."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--py", type=str,
                   help="Expr in x,y. Example: \"x + y - y**2*np.exp(x)\"")
    g.add_argument("--py-func", type=str,
                   help="Code that defines def f(x, y): return ...")
    g.add_argument("--py-file", type=str,
                   help="Path to a Python file that defines def f(x, y)")

    p.add_argument("--exact", type=str, default=None,
                   help="Expr of exact y(x). Use numpy ops if possible")

    p.add_argument("-a", type=float, required=True, help="Left bound a")
    p.add_argument("-b", type=float, required=True, help="Right bound b")
    p.add_argument("-N", type=int, required=True, help="Number of steps")
    p.add_argument("--theta", type=float, required=True,
                   help="For forward: y(a)=theta. For backward: y_{N-1}=theta at x=b-h")

    p.add_argument("--method", choices=list(SOLVERS.keys()),
                   default="euler_forward", help="Numerical method")
    p.add_argument("--print-first", type=int, default=10, help="How many points to print")
    p.add_argument("--out", type=str, default="", help="Optional CSV output path")
    p.add_argument("--no-header", action="store_true", help="Skip header line")
    p.add_argument("--plot", action="store_true", help="Show a matplotlib plot")
    p.add_argument("--save-plot", type=str, default="", help="Save plot to a PNG path")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Build f
    if args.py is not None:
        f = build_f_from_py_expr(args.py)
        f_title = args.py
    elif args.py_func is not None:
        f = build_f_from_py_func(args.py_func)
        f_title = "f from --py-func"
    else:
        f = build_f_from_file(args.py_file)
        f_title = f"f from {args.py_file}"

    # Build exact if given
    y_exact = None
    if args.exact:
        y_exact = build_y_exact_from_expr(args.exact)

    # Solve
    if args.method == "euler_forward":
        X, Y = euler_forward(f, args.a, args.b, args.theta, args.N)
        endpoint_report = f"y(b) approx = {Y[-1]:.12g}"
    else:
        X, Y = euler_backward(f, args.a, args.b, args.theta, args.N)
        endpoint_report = f"y(a) approx = {Y[0]:.12g}"

    # Print table
    k = min(args.print_first, len(X))
    if not args.no_header:
        print("i,x,y_approx")
    for i in range(k):
        print(f"{i},{X[i]:.12g},{Y[i]:.12g}")
    if k < len(X):
        print(f"... total {len(X)} points")
    print(endpoint_report)

    # Optional CSV
    if args.out:
        try:
            with open(args.out, "w", newline="", encoding="utf-8") as fcsv:
                w = csv.writer(fcsv)
                w.writerow(["x", "y"])
                for xi, yi in zip(X, Y):
                    w.writerow([f"{xi:.12g}", f"{yi:.12g}"])
            print("Saved:", args.out)
        except Exception as e:
            print("Could not save CSV:", e, file=sys.stderr)
            return 2

    # Plot
    if args.plot or args.save_plot:
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(X, Y, marker="o", label=args.method)
            if y_exact is not None:
                try:
                    Yex = y_exact(X)
                except Exception:
                    Yex = np.array([y_exact(float(xi)) for xi in X], dtype=float)
                plt.plot(X, Yex, label="Exact")
                title = f"y' = {f_title} + Exact"
            else:
                title = f"y' = {f_title}"
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.legend()
            if args.save_plot:
                fig.savefig(args.save_plot, dpi=150, bbox_inches="tight")
                print("Saved plot:", args.save_plot)
            if args.plot:
                plt.show()
            else:
                plt.close(fig)
        except Exception as e:
            print("Plotting failed:", e, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
