# HiC2MicroC

## Install

The following Python libraries are required. 

    1. Cooler (https://cooler.readthedocs.io/en/latest/index.html)
    2. Pytorch 
    3. NumPy
    4. Pandas
    5. SciPy
    6. einops (https://einops.rocks/) 


## Usage 

Three parameters required:

python HiC2MicroC.py -f1 file1 -f2 file2 -f3 file3

1. file1: a 5-kb Hi-C file in cool format
1. file2: chromosome size file
1. file3: output prefix

HiC2MicroC will generate file3.cool, which is our predicted Micro-C file. 
