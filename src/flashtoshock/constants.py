'''
Constants File for Flash to Shock
'''

import math
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import brentq
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Flash-to-Shock Analysis", layout="wide")

# --- Physical and Model Constants ---
ftm = 0.3048     # feet to meters
ktj = 4.184e+12  # kt to Joules
P0  = 101325.0   # Sea-level atmospheric pressure (Pa)
H   = 8000.0     # Atmospheric scale height (m)
R_s = 287.05     # Specific gas constant for dry air (J/kg*K)
gam = 1.4        # Adiabatic index for air

# Wei-Hargather weak shock coefficients
u, v, w, p, q = 0.45, 1.0, 15.0, 1.4, 0.3