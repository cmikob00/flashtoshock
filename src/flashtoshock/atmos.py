'''
Atmospheric Model for Flash to Shock
'''

from flashtoshock.constants import *

# --- Core Physics & Atmosphere ---
# NOTE: Physics updated to handle terrain elevation.

def atmos_properties_at_alt(alt_msl, T_sensor_c, sensor_alt_msl=0.0):
    """Calculates atmospheric properties tracking MSL pressure and AGL temp lapse."""
    T_sensor_k = T_sensor_c + 273.15
    L = 0.0065  # Standard tropospheric temperature lapse rate (K/m)
    
    # Temperature lapses based on height ABOVE the sensor
    h_above_sensor = alt_msl - sensor_alt_msl
    T_k = max(216.65, T_sensor_k - (L * h_above_sensor))
    
    # Pressure decays based on absolute MSL altitude
    P = P0 * math.exp(-alt_msl / H)
    
    # Calculate rho strictly via the Ideal Gas Law
    rho = P / (R_s * T_k)
    
    a = np.sqrt(gam * R_s * T_k)  
    return P, T_k, rho, a