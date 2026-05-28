'''
HOB Solver for Flash to Shock
'''

from flashtoshock.constants import *
from flashtoshock.atmos import *

def get_shock_arrival_time(hob, W_kt, T_ground_c, n_layers=100, shock_model='weak', terr_alt_m=0.0):
    """Calculates arrival time for a vertical path from HOB (AGL) to a sensor at terr_alt_m (MSL)."""
    if hob <= 1.0: return 1e6
    W_j = W_kt * ktj
    dh = hob / n_layers  # Layer thickness is in AGL coordinates
    total_time = 0.0
    for i in range(n_layers):
        # The midpoint of the layer in AGL coordinates (distance from the ground)
        z_mid_agl = (i + 0.5) * dh
        # The absolute MSL altitude of that layer
        z_mid_msl = z_mid_agl + terr_alt_m
        
        # Distance from the burst point (which is at hob AGL) down to the layer midpoint
        r_from_burst = hob - z_mid_agl
        
        # Get atmospheric properties at the layer's true MSL altitude
        _, _, rho, a = atmos_properties_at_alt(z_mid_msl, T_ground_c, sensor_alt_msl=terr_alt_m)
        
        # Wei & Hargather (2023) Eq 6
        lc = (W_j / (rho * a**2))**(1.0 / 3.0) 
        
        rstar_target = r_from_burst / lc
        if shock_model == 'strong':
            tstar_sol = rstar_target**(2.5)
            U_star = 0.4 * (tstar_sol**(-0.6)) if tstar_sol >= 1e-9 else 1.0
        else:
            def rstar_residual(ts):
                if ts <= 0: return rstar_target
                return ts + u * (np.log(v + w * ts**p))**q - rstar_target
            try:
                tstar_sol = brentq(rstar_residual, 1e-9, rstar_target + 1.0)
            except ValueError:
                tstar_sol = rstar_target
            if tstar_sol < 1e-9:
                 U_star = 1.0
            else:
                log_val = np.log(v + w * tstar_sol**p)
                U_star = 1.0 + (u*q*(log_val)**(q-1) * (1.0/(v+w*tstar_sol**p)) * w*p*tstar_sol**(p-1))
        U_shock = a * U_star
        total_time += dh / U_shock
    return total_time

def find_hob_for_time(target_time, W_kt, T_ground_c, shock_model='weak', terr_alt_m=0.0):
    """Finds HOB (AGL) for a sensor at a given terrain elevation (MSL)."""
    def time_residual(hob):
        return get_shock_arrival_time(hob, W_kt, T_ground_c, shock_model=shock_model, terr_alt_m=terr_alt_m) - target_time
    try:
        return brentq(time_residual, 1.1, 50000.0) 
    except ValueError:
        return np.nan