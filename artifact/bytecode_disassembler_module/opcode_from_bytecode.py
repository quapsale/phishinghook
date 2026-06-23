from evmdasm.evmdasm import EvmBytecode
import pandas as pd
import os
from numpy import loadtxt

bytecodes = loadtxt('path/to/contract/bytecodes.txt', dtype=str).tolist()
opcodes = []

for i, bytecode in enumerate(bytecodes):
    print('Decoding {0} out of {1}'.format(i+1, len(bytecodes)))
    address = i
    evmcode = EvmBytecode(bytecode)
    evminstructions = evmcode.disassemble()
    
    for instr in evminstructions:
        opcode = {'Contract': address, 'Opcode': instr.name, 'Operand': instr.operand, 'Gas': instr.gas, 'Label': 1}
        opcodes.append(opcode)

df = pd.DataFrame.from_records(opcodes, columns=['Contract', 'Opcode', 'Operand', 'Gas', 'Label'])

if os.path.isfile('opcodes.csv'):
     df2 = pd.read_csv('opcodes.csv')
     df = df2._append(df)

df.to_csv('opcodes.csv', index=False)
