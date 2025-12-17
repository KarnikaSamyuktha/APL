"""Tiny combinational logic simulator producing WaveDrom JSON.

Usage:
  python digitalsim.py path/to/circuit.net [--out out.json]

Reads a simple digital circuit netlist (INPUTS, OUTPUTS, GATES, STIMULUS),
simulates its behavior for given stimuli, and outputs WaveDrom-compatible
JSON showing signal waveforms.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List

def parse_netlist(text: str):
    """Parse the text content of the .net file into sections."""
    # Remove blank lines and comments
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]

    # Predefined sections of the file
    sections = {"INPUTS": [], "OUTPUTS": [], "GATES": [], "STIMULUS": []}
    valid_sections = set(sections.keys())
    curr_section = None

    # Read line by line, filling each section
    for line in lines:
        if ":" in line and (not line.endswith(":")):
            # Handle lines like: INPUTS: A B C, OUTPUTS: Y
            curr_section, val = [part.strip() for part in line.split(":")]
            if curr_section not in valid_sections:
                raise ValueError(f"Unknown section name: {curr_section}")
            sections[curr_section] = val.strip().split()
        elif line.endswith(":"):
            # Handle lines like: GATES:\n , STIMULUS:\n
            curr_section = line.split(":")[0].strip()
            if curr_section not in valid_sections:
                raise ValueError(f"Unknown section name: {curr_section}")
        else:
            # Regular content line within a section
            if curr_section is None:
                raise ValueError(f"Line found outside of a section: {line}")
            sections[curr_section].append(line)

    # Validation: ensure all required sections are present and non-empty
    for sec in ["INPUTS", "OUTPUTS", "GATES", "STIMULUS"]:
        if not sections.get(sec):
            raise ValueError(f"Missing or empty section: {sec}")

    return sections


def eval_gate(Input_values: dict, gate_list: dict):
    """Evaluate all gates based on input signals and their logic."""
    gate_values = {}
    valid_gates = {"AND", "OR", "NOT", "XOR"}

    for gate, values in gate_list.items():
        gate_values[gate] = []

        # Extract gate type , Check for Unknown gates
        gate_type = values.split("(")[0].strip()
        if gate_type not in valid_gates:
            raise ValueError(f"Unknown gate type: {gate_type}")

        # Each block handles a specific gate type
        if "NOT" in values:
            # Unary gate
            input = values[values.find("(")+1 : values.find(")")].strip()
            for i in range(len(Input_values[input])):
                gate_values[gate].append(int(not Input_values[input][i]))

        elif "OR" in values:
            input = [v.strip() for v in values[values.find("(")+1 : values.find(")")].split(",")]
            if len(input) != 2:
                raise ValueError(f"OR gate '{gate}' must have exactly 2 inputs.")
            for i in range(len(Input_values[input[0]])):
                gate_values[gate].append(Input_values[input[0]][i] or Input_values[input[1]][i])

        elif "AND" in values:
            input = [v.strip() for v in values[values.find("(")+1 : values.find(")")].split(",")]
            if len(input) != 2:
                raise ValueError(f"AND gate '{gate}' must have exactly 2 inputs.")
            for i in range(len(Input_values[input[0]])):
                gate_values[gate].append(Input_values[input[0]][i] and Input_values[input[1]][i])

        elif "XOR" in values:
            input = [v.strip() for v in values[values.find("(")+1 : values.find(")")].split(",")]
            if len(input) != 2:
                raise ValueError(f"XOR gate '{gate}' must have exactly 2 inputs.")
            for i in range(len(Input_values[input[0]])):
                gate_values[gate].append(Input_values[input[0]][i] ^ Input_values[input[1]][i])

        # Update the dictionary so newly created signals can be reused
        Input_values.update(gate_values)
    return Input_values


def simulate(data: dict):
    """Run the simulation using parsed netlist data."""
    # Initialize dictionary for all input signals
    Input_values = {var: [] for var in data["INPUTS"]}
    input_list = list(Input_values.keys())
    time = []

    # Parse stimulus lines (time + input values)
    for dat in data["STIMULUS"]:
        parts = dat.split()
        t = int(parts[0])
        time.append(t)
        #CHECK if STImMULUS has all inputs data
        if len(parts) - 1 != len(input_list):
            raise ValueError(f"Stimulus line has wrong number of inputs: {dat}")
        for i in range(1, len(parts)):
            Input_values[input_list[i - 1]].append(int(parts[i]))

    # Validate that time points are strictly increasing
    if time != sorted(set(time)):
        raise ValueError("STIMULUS times are not strictly increasing or contain duplicates.")

    # Build the dictionary of gates and their definitions
    gate_list = {}
    known_signals = set(data["INPUTS"])

    for gate in data["GATES"]:
        if "=" not in gate:
            raise ValueError(f"Invalid gate line: {gate}")
        key, val = gate.split("=")
        key, val = key.strip(), val.strip()
        gate_type = val.split("(")[0].strip()
        inputs = [x.strip() for x in val[val.find("(")+1 : val.find(")")].split(",")]

        # Ensure all inputs are defined before use
        for inp in inputs:
            if inp not in known_signals:
                raise ValueError(f"Unknown signal '{inp}' used in gate '{gate}'")

        known_signals.add(key)
        gate_list[key] = val

    # Evaluate all gates and produce final signal values
    result = eval_gate(Input_values, gate_list)
    return result


def to_wavedrom_json(sections: dict, values: dict):
    """Convert simulated signal values into WaveDrom-compatible JSON."""
    WaveDrom = {"signal": []}
    for key, value in values.items():
        if key in sections["INPUTS"]:
            # Input signal waveform
            WaveDrom["signal"].append({"name": key, "wave": "".join(str(v) for v in value)})
        elif key in sections["OUTPUTS"]:
            # Output signal waveform
            WaveDrom["signal"].append({"name": key, "wave": "".join(str(v) for v in value)})
    return WaveDrom


def main(argv: List[str]) -> int:
    """Command-line entry point: parse, simulate, and write JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("netlist", help=".net file path")
    ap.add_argument("--out", "-o", help="output JSON path")
    args = ap.parse_args(argv)

    # Read input file with try and except block
    try:
        text = Path(args.netlist).read_text()
    except FileNotFoundError:
        print(f"Error: file '{args.netlist}' not found.")
        return 1

    # Parse -> Simulate -> Convert to JSON
    nl = parse_netlist(text)
    waves = simulate(nl)
    js = to_wavedrom_json(nl, waves)

    #output path
    out_path = args.out
    if not out_path:
        p = Path(args.netlist)
        out_path = str(p.with_suffix(".json"))

    # Write WaveDrom JSON to file
    Path(out_path).write_text(json.dumps(js,indent=1) + "\n")
    print(out_path)
    return 0


# Program Execution
if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
