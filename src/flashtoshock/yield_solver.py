'''
Yield Solver for Flash to Shock
'''

from flashtoshock.constants import *

def find_yield_for_measurement(d_meas, t_meas, rho, a, shock_model='weak'):
    if t_meas <= 0 or d_meas <= 0: return np.nan
    if shock_model == 'strong':
        try:
            # Dimensionless ratio r* / t* = d / (a*t)
            ratio = d_meas / (a * t_meas)
            tstar = ratio**(-5.0 / 3.0)
        except (ValueError, ZeroDivisionError):
            return np.nan
    else: 
        def residual(tstar):
            if tstar <= 0: return 1e6
            model_rstar = tstar + u * (np.log(v + w * tstar**p))**q
            # Dimensionless ratio applied to weak shock model
            measurement_rstar = tstar * d_meas / (a * t_meas)
            return model_rstar - measurement_rstar
        try:
            tstar = brentq(residual, 1e-9, 1e6)
        except ValueError:
            return np.nan
            
    # Wei & Hargather (2023) Eq 7 logic inverted
    tc = t_meas / tstar
    lc = tc * a
    return ((lc**3) * rho * (a**2)) / ktj