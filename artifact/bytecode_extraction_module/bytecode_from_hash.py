import requests
from numpy import loadtxt

ETHERSCAN_API_KEY = 'YOUR_KEY_HERE'

addresses = loadtxt('path/to/contract/hashes.txt', dtype=str).tolist()

with open('path/to/contract/bytecodes.txt', 'a+') as f:        
    for address in addresses:
        url = f'https://api.etherscan.io/api?module=proxy&action=eth_getCode&address={address}&tag=latest&apikey={ETHERSCAN_API_KEY}'
        response = requests.get(url)
        data = response.json()
        bytecode = data['result']
        print('Decoded address {0} out of {1}: hash {2}'.format(addresses.index(address)+1, len(addresses), address))
        f.write('%s\n' %bytecode)
