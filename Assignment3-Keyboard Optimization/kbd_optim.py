"""
Keyboard Layout Optimization via Simulated Annealing

Notes:
- Cost is total Euclidean distance between consecutive characters.
- Coordinates are fixed (QWERTY-staggered grid). Optimization swaps assignments.

This base code uses Python "types" - these are optional, but very helpful
for debugging and to help with editing.

"""
import argparse 
import json
import math
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore

Point = Tuple[float, float] #defining coordinates for characters
Layout = Dict[str, Point] #assigning characters to points


def qwerty_coordinates(chars: str) -> Layout:
    """Return QWERTY grid coordinates for the provided character set.

    The grid is a simple staggered layout (units are arbitrary):
    - Row 0: qwertyuiop at y=0, x in [0..9]
    - Row 1: asdfghjkl at y=1, x in [0.5..8.5]
    - Row 2: zxcvbnm at y=2, x in [1..6]
    - Space at (4.5, 3)
    Characters not present in the grid default to the space position.
    """
    row0 = "qwertyuiop"
    row1 = "asdfghjkl"
    row2 = "zxcvbnm"

    coords: Layout = {}
    for i, c in enumerate(row0):
        coords[c] = (float(i), 0.0) #row0
    for i, c in enumerate(row1):
        coords[c] = (0.5 + float(i), 1.0) #row1
    for i, c in enumerate(row2):
        coords[c] = (1.0 + float(i), 2.0) #row2
    coords[" "] = (4.5, 3.0) #row3

    # Backfill for requested chars; unknowns get space position.
    space_xy = coords[" "]
    for ch in chars:
        if ch not in coords:
            coords[ch] = space_xy
    return coords


def initial_layout() -> Layout:
    """Create an initial layout mapping chars to some arbitrary positions of letters."""

    # Start with identity for letters and space; others mapped to space.
    base_keys = "abcdefghijklmnopqrstuvwxyz "

    # Get coords - or use coords of space as default
    layout = qwerty_coordinates(base_keys)
    return layout


def preprocess_text(text: str, chars: str) -> str:
    """Lowercase and filter to the allowed character set; map others to space."""
    text=text.lower() #convert text to lowercase
    allowed=[] #list of allowed characters
    for ch in text:
        if ch in chars:
            allowed.append(ch)
        else:
            allowed.append(" ") #keyspace for other characters
    processed_text="".join(allowed) #convert list to string
    return processed_text

def path_length_cost(text: str, layout: Layout) -> float:
    """Sum Euclidean distances across consecutive characters in text."""
    distance=0
    i=0
    while i<(len(text)-1):
        """iterate through consecutive chars for the given text
        calculate distance between the respective keys for chars in (x,y) coordinates for the given layout
        add to the total distance"""
        distance+=((layout[text[i+1]][0]-layout[text[i]][0])**2 + (layout[text[i+1]][1]-layout[text[i]][1])**2)**0.5
        i+=1
    return distance

# Dataclass is like a C struct - you can use it just to store data if you wish
# It provides some convenience functions for assignments, printing etc.
@dataclass
class SAParams:
    iters: int = 50000 #total no:of iterations
    t0: float = 1.0  # Initial temperature setting
    alpha: float = 0.999  # geometric decay per iteration
    epoch: int = 1  # iterations per temperature step (1 = per-iter decay)


def simulated_annealing(
    text: str,
    layout: Layout,
    params: SAParams,
    rng: random.Random,
) -> Tuple[Layout, float, List[float], List[float]]:
    """Simulated annealing to minimize path-length cost over character swaps.

    Returns best layout, best cost, and two lists:
    - best cost up to now (monotonically decreasing)
    - cost of current solution (may occasionally go up)
    These will be used for plotting
    """
    best_layout=layout 
    best_cost=path_length_cost(text,best_layout)
    current_layout=best_layout.copy()
    current_cost=best_cost
    keys = list(layout.keys()) #list of keys(chars)
    T=params.t0 #initial temperature
    bestcost_list=[]
    bestcost_list.append(best_cost) #list of best costs
    currentcost_list=[]
    currentcost_list.append(current_cost)#list of current costs
    for count in range(params.iters):
        k1,k2=rng.sample(keys,2) #take any two random chars
        new_layout=current_layout.copy()
        new_layout[k1],new_layout[k2]=new_layout[k2],new_layout[k1] #swap the random chars in the new layout
        new_cost = path_length_cost(text, new_layout) #calculate path length for new layout
        delta=new_cost-current_cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            #if (new_cost < current_cost) or exp(-delta/current Temperature)>(0,1) replace the current layout
            current_layout = new_layout
            current_cost = new_cost
            currentcost_list.append(current_cost)
            if new_cost < best_cost: 
                best_layout = new_layout
                best_cost = new_cost
                bestcost_list.append(best_cost)
        T=(params.alpha)*T #geometric decay of temperature
    return (best_layout,best_cost,bestcost_list,currentcost_list)

def plot_costs(
    layout: Layout, best_trace: List[float], current_trace: List[float]
) -> None:

    # Plot cost trace
    out_dir = "."
    plt.figure(figsize=(6, 3))
    plt.plot(best_trace, lw=1.5)
    plt.plot(current_trace, lw=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("Best Cost vs Iteration")
    plt.tight_layout()
    path = os.path.join(out_dir, f"cost_trace.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Plot layout scatter
    xs, ys, labels = [], [], []
    for ch, (x, y) in layout.items():
        xs.append(x)
        ys.append(y)
        labels.append(ch)

    plt.figure(figsize=(6, 3))
    plt.scatter(xs, ys, s=250, c="#1f77b4")
    for x, y, ch in zip(xs, ys, labels):
        plt.text(
            x,
            y,
            ch,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="#1f77b4", ec="none", alpha=0.9),
        )
    plt.gca().invert_yaxis()
    plt.title("Optimized Layout")
    plt.axis("equal")
    plt.tight_layout()
    path = os.path.join(out_dir, f"layout.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def load_text(filename) -> str:
    #if file input given
    if filename is not None:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback demo text:)
    return (
        "the quick brown fox jumps over the lazy dog\n"
        "APL is the best course ever\n"
    )


def main(filename: str | None = None) -> None:
    rng = random.Random(0)
    chars = "abcdefghijklmnopqrstuvwxyz "

    #for user-inputs (Using arg parse module)
    parser = argparse.ArgumentParser()
    #define names and conditions of every arguement
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--iters", type=int, default=50000)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.999)
    args = parser.parse_args()

    # Initial assignment - QWERTY
    layout0 = initial_layout()

    # Prepare text and evaluate baseline
    raw_text = load_text(args.input)#input file here
    text = preprocess_text(raw_text, chars)

    baseline_cost = path_length_cost(text, layout0)
    print(f"Baseline (QWERTY assignment) cost: {baseline_cost:.4f}")

    # Annealing - give parameter values
    params = SAParams(iters=args.iters, t0=args.t0, alpha=args.alpha)#input parameters here
    start = time.time()
    best_layout, best_cost, best_trace, current_trace = simulated_annealing(text,layout0, params, rng)
    dur = time.time() - start #calculate run time
    print(f"Optimized cost: {best_cost:.4f}  (improvement {(baseline_cost - best_cost):.4f})")
    print(f"Runtime: {dur:.2f}s over {params.iters} iterations")

    plot_costs(best_layout, best_trace, current_trace)


if __name__ == "__main__":
    main()
