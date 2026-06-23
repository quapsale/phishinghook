# PhishingHook: Catching Phishing Ethereum Smart Contracts Leveraging EVM Opcodes

[![DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.14260284-blue.svg)](https://doi.org/10.5281/zenodo.14260284)
[![Paper DOI](https://img.shields.io/badge/IEEE_DSN_2025-10.1109%2FDSN64029.2025.00033-purple.svg)](https://doi.org/10.1109/DSN64029.2025.00033)

## 📌 Overview
PhishingHook is an end-to-end framework for extracting, disassembling, and analyzing Ethereum smart contract bytecode to identify malicious phishing contracts. By evaluating contract mechanics through EVM opcodes, it provides multiple detection methodologies spanning statistical classifiers, language models, and vision-based architectures.

This repository serves as the official implementation for the paper: *"PhishingHook: Catching Phishing Ethereum Smart Contracts leveraging EVM Opcodes"* (IEEE DSN 2025).

---

## 🗄️ Dataset Availability
The complete dataset used to evaluate this framework includes extensive raw smart contract bytecodes, structured features, and labeled benign/phishing metadata. 

Due to its size, the dataset is permanently archived and publicly accessible via Zenodo:
👉 **[Download the Dataset on Zenodo](https://doi.org/10.5281/zenodo.14260284)**

> 💡 **Setup Note:** If running the evaluation or training pipelines locally, please download the dataset archive from Zenodo and unpack it into a local `/data` directory at the root of this project.

---

## 🚀 Quick Start & Installation (Core Modules)

### Prerequisites
* **Python:** 3.10.12
* **R:** 4.4.2 (Required only for post-hoc statistical analysis scripts)

### Installation Steps
```bash
# Clone the repository
git clone git@github.com:quapsale/phishinghook.git
cd phishinghook

# Set up and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
*Note: R dependencies required for post-hoc validation (e.g., for `kruskal_wallis.R`) are listed in `requirements.txt` and should be installed in your local R environment.*

---

## 🐳 Dockerized Command Line Interface (CLI)
For rapid onboarding and reproducibility, this project includes a Dockerized CLI for analyzing Ethereum smart contract bytecode. It provides two core functionalities:

1. **Disassembly**: Converts raw EVM bytecode into a structured sequence of instructions.
2. **Phishing Detection**: Applies a pretrained Random Forest machine learning model to classify smart contracts as either phishing or legitimate.

### Docker Setup
Ensure you have [Docker](https://www.docker.com/) installed on your host system, then build the image:
```bash
docker build -t phishinghook .
```

### Usage: Disassemble Bytecode
Converts raw EVM bytecode into a CSV file listing each instruction with its corresponding opcode, operand, and gas estimate.
```bash
docker run --rm -v "$PWD:/execution" phishinghook disassemble '0x6080604052600036606060008073afe266e64e373ae7e74aa5ef0ce417198b3aa4c96001600160a01b03168585604051603892919060ab565b600060405180830381855af49150503d80600081146071576040519150601f19603f3d011682016040523d82523d6000602084013e6076565b606091505b509150915081609e5760405162461bcd60e51b815260040160959060bb565b60405180910390fd5b8051945060200192505050f35b6000828483379101908152919050565b60208082526004908201526311985a5b60e21b60408201526060019056fea2646970667358221220e9e59d6a9f0befcfdf64f9baf7d5d970bad8fb28aee9fde6b53a300b470c5c6064736f6c63430008000033'
```
* **Output:** A CSV file with columns: `InstructionIndex`, `OPCODE`, `OPERAND`, `GAS`.

### Usage: Detect Phishing Contracts
Analyzes bytecode using a pretrained Histogram Similarity Classifier (Random Forest) to determine whether the contract is malicious.
```bash
docker run --rm -v "$PWD:/execution" phishinghook detect '0x6080604052600036606060008073afe266e64e373ae7e74aa5ef0ce417198b3aa4c96001600160a01b03168585604051603892919060ab565b600060405180830381855af49150503d80600081146071576040519150601f19603f3d011682016040523d82523d6000602084013e6076565b606091505b509150915081609e5760405162461bcd60e51b815260040160959060bb565b60405180910390fd5b8051945060200192505050f35b6000828483379101908152919050565b60208082526004908201526311985a5b60e21b60408201526060019056fea2646970667358221220e9e59d6a9f0befcfdf64f9baf7d5d970bad8fb28aee9fde6b53a300b470c5c6064736f6c63430008000033'
```
* **Output:** A printed label: `Phishing Detected` OR `Legitimate Smart Contract`.

### Model Format Note
If providing a custom model for the detection CLI, the model file must be a Python pickle (`.pkl`) file containing a dictionary with two keys:
```python
{
  'model': <sklearn.ensemble.RandomForestClassifier>,
  'all_opcodes': <list of opcode strings used as features>
}
```

---

## 📂 Modules & Repository Structure

### 1. Bytecode Disassembler (`/evmdasm`, `opcode_from_bytecode.py`)
Handles parsing raw EVM bytecode into readable assembly language. Uses an updated, customized implementation of the `evmdasm` library optimized for the **Shanghai fork**.
* `opcode_from_bytecode.py`: Streamlined pipeline to decode bytecodes and save processed opcodes directly to a clean CSV.

### 2. Bytecode Extraction (`bytecode_from_hash.py`)
Infrastructure utility to fetch raw bytecode directly from the blockchain via the Etherscan API using smart contract addresses.

### 3. Model Evaluation (`/models`, `/scalability`, `/time_resistance`)
Implements and benchmarks the various model architectures detailed in the paper:
* **Histogram Similarity Classifiers:** Frequency-based statistical baselines.
* **Language Models:** Implementations of sequence-aware models including SCSGuard, GPT-2, and T5.
* **Vision Models:** Bytecode-to-image pipeline leveraging ViT+R2D2, ViT+Freq, and ECA+EfficientNet.
* **Vulnerability Detection Baselines:** Integrates the ESCORT model for cross-functional security assessment.
* Includes localized modules for horizontal scalability mapping and temporal data-drift validation (`/time_resistance`).

### 4. Post-Hoc Statistical Analysis (`/post_hoc`)
Contains the rigorous evaluation logic used to validate performance gains:
* `kruskal_wallis.R`: R script executing Shapiro-Wilk normality checking, Kruskal-Wallis variance testing, and Dunn's post-hoc pairwise comparisons.

---

## 📄 Citation
If you use this framework, dataset, or find our methodology helpful in your security research, please cite our IEEE DSN 2025 paper:

```bibtex
@INPROCEEDINGS{11068866,
  author={De Rosa, Pasquale and Queyrut, Simon and Bromberg, Yérom-David and Felber, Pascal and Schiavoni, Valerio},
  booktitle={2025 55th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)}, 
  title={PhishingHook: Catching Phishing Ethereum Smart Contracts leveraging EVM Opcodes}, 
  year={2025},
  volume={},
  number={},
  pages={222-232},
  keywords={Codes;Accuracy;Phishing;Smart contracts;Decentralized applications;Performance metrics;Virtual machines;Reproducibility of results;Blockchains;Research and development;EVM;opcodes;smart contracts;phishing;detection},
  doi={10.1109/DSN64029.2025.00033}}
```
