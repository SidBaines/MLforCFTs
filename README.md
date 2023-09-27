# MLforCFTs

## Setup
To get the environment set up, create a venv using the requirements.txt file:
```bash
python3 -m venv <YourEnvironmentName>
source <YourEnvironmentName>/bin/activate
python3 -m pip install -r requirements.txt
```
You'll also need to make sure that you have julia installed and available for pysr to work properly, and to install the Julia dependencies. For that, follow the instructions in the PySR github [here](https://github.com/MilesCranmer/PySR#pip).

## Running
I recommend running interactively, which will give you the freedom to change/test things afterwards, and importantly to do something if the pysr package just doesn't build for some reason.
Then, to run, use the file ```SymbolicRegressionExample.py```:
```bash
# Make sure you have <YourEnvironmentName> activated!
python3 -i SymbolicRegressionExample.py
```