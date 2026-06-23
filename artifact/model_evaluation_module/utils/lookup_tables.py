import pandas as pd
import json

# Load the CSV file
file_path = 'opcodes.csv'
df = pd.read_csv(file_path)

# List all existing opcodes and operands
opcodes = df['Opcode'].value_counts()
operands = df['Operand'].value_counts(dropna=False)
gas = df['Gas'].value_counts()

# Rank them by frequency of appearance in reverse order (most frequent last)
opcode_lookup = {i: opcode for i, opcode in enumerate(opcodes.index[::-1])}
operand_lookup = {i: operand for i, operand in enumerate(operands.index[::-1])}
gas_lookup = {i: gas for i, gas in enumerate(gas.index[::-1])}

# Ensure the keys range from 0 to 255 using normalization
def normalize_keys(lookup):
    keys = list(lookup.keys())
    min_key = min(keys)
    max_key = max(keys)
    normalized_lookup = {((key - min_key) / (max_key - min_key)) * 255: value for key, value in lookup.items()}
    return normalized_lookup

opcode_lookup = normalize_keys(opcode_lookup)
operand_lookup = normalize_keys(operand_lookup)
gas_lookup = normalize_keys(gas_lookup)

# Save lookup tables to JSON files
with open('opcode_lookup.json', 'w') as f:
    json.dump(opcode_lookup, f)

with open('operand_lookup.json', 'w') as f:
    json.dump(operand_lookup, f)

with open('gas_lookup.json', 'w') as f:
    json.dump(gas_lookup, f)

# Print the first and last elements of the lookup tables
print("First Opcode Lookup Table Element:", list(opcode_lookup.items())[0])
print("Last Opcode Lookup Table Element:", list(opcode_lookup.items())[-1])
print("First Operand Lookup Table Element:", list(operand_lookup.items())[0])
print("Last Operand Lookup Table Element:", list(operand_lookup.items())[-1])
print("First Gas Lookup Table Element:", list(gas_lookup.items())[0])
print("Last Gas Lookup Table Element:", list(gas_lookup.items())[-1])
