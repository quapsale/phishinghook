#!/usr/bin/env python3

import argparse
import pickle
from collections import Counter
import numpy as np
import pandas as pd
from evmdasm.evmdasm import EvmBytecode

def print_banner():
    """Print ASCII banner and welcome message for PhishingHook."""
    banner = r"""
______ _     _     _     _             _   _             _    
| ___ \ |   (_)   | |   (_)           | | | |           | |   
| |_/ / |__  _ ___| |__  _ _ __   __ _| |_| | ___   ___ | | __
|  __/| '_ \| / __| '_ \| | '_ \ / _` |  _  |/ _ \ / _ \| |/ /
| |   | | | | \__ \ | | | | | | | (_| | | | | (_) | (_) |   < 
\_|   |_| |_|_|___/_| |_|_|_| |_|\__, \_| |_/\___/ \___/|_|\_\
                                  __/ |                       
                                 |___/   
    """
    print(banner)
    print("Welcome to PhishingHook, the first machine learning-based detector of phishing EVM contracts.\n")


def disassemble_bytecode(bytecode: str, out_file: str):
    """
    Disassemble the given bytecode string and save the instructions to out_file (CSV).
    """
    evmcode = EvmBytecode(bytecode)
    instructions = evmcode.disassemble()

    records = []
    for idx, instr in enumerate(instructions):
        records.append({
            "InstructionIndex": idx,
            "OPCODE": instr.name,
            "OPERAND": instr.operand,
            "GAS": instr.gas
        })

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(records, columns=["InstructionIndex", "OPCODE", "OPERAND", "GAS"])
    df.to_csv(out_file, index=False)

    print(f"Disassembly complete. Instructions saved to: {out_file}\n")


def build_feature_vector_from_bytecode(bytecode: str, all_opcodes: list) -> np.ndarray:
    """
    Given a single contract bytecode and the master list of opcodes used during training,
    returns a 1D NumPy array representing the histogram of instructions 
    in the same order as `all_opcodes`.
    """
    evmcode = EvmBytecode(bytecode)
    instructions = evmcode.disassemble()
    opcode_names = [instr.name for instr in instructions]

    opcode_counts = Counter(opcode_names)
    feature_vector = [opcode_counts.get(opc, 0) for opc in all_opcodes]
    return np.array(feature_vector).reshape(1, -1)


def detect_phishing(bytecode: str):
    """
    Loads a pre-trained Random Forest model from a fixed path,
    disassembles the bytecode, builds features, and classifies.
    """
    MODEL_PATH = 'random_forest_model.pkl'
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        all_opcodes = data['all_opcodes']

    feature_vector = build_feature_vector_from_bytecode(bytecode, all_opcodes)

    prediction = model.predict(feature_vector)[0]

    if prediction == 1:
        print("Phishing Detected!\n"
        "This smart contract exhibits characteristics commonly associated with phishing attacks.\n"
        "Proceed with extreme caution before interacting or signing any transactions.\n")
    else:
        print("Legitimate Smart Contract.\n"
            "No phishing patterns were detected in the provided bytecode.\n"
            "The contract appears safe based on current analysis.\n")



def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="PhishingHook – disassemble EVM bytecode or detect phishing using a pretrained Random Forest."
    )
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    # -------------------------
    # Subcommand: disassemble
    # -------------------------
    dis_parser = subparsers.add_parser(
        'disassemble',
        help='Disassemble the provided EVM bytecode and save the output to a CSV file.'
    )
    dis_parser.add_argument('bytecode', type=str,
                            help='The EVM bytecode string to disassemble.')

    # -------------------------
    # Subcommand: detect
    # -------------------------
    detect_parser = subparsers.add_parser(
        'detect',
        help='Use a pretrained Random Forest model to detect if the contract is phishing.'
    )
    detect_parser.add_argument('bytecode', type=str,
                               help='The EVM bytecode string to analyze.')

    args = parser.parse_args()

    if args.command == 'disassemble':
        out_file = 'disassembled.csv'
        disassemble_bytecode(args.bytecode, out_file)

    elif args.command == 'detect':
        detect_phishing(args.bytecode)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
