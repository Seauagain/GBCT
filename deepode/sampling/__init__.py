## system import 
import cantera as ct 
import sys

## custom import 
from .CanteraTools import *
from .SampleMethod import *
from .DataProcess  import DataProcess
from .utils_data import *
from .VisualArt import VisualArt


def setMechanism(mech_path):
    gas = ct.Solution(mech_path)
    for filename in ['', '.CanteraTools', '.SampleMethod']:
        ## convert to modules
        mod = sys.modules[__name__ + filename]
        ## set global attribute
        setattr(mod, 'gas', gas)
    # print(f"choose mechanism: {mech_path}")
