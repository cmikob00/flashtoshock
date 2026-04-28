# Flash to Shock Analysis Streamlit App
# Adapted by AI, 21 April 2026
# Unified application for determining Yield or Height of Burst.
# UNCLASSIFIED

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

# --- Helper Functions ---
def parse_input(val_str):
    """Safely converts strings to floats, parsing simple fractions if present."""
    try:
        if '/' in val_str:
            num, den = val_str.split('/')
            return float(num) / float(den)
        return float(val_str)
    except Exception:
        raise ValueError("Invalid numerical input")

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

# --- Plotting Functions ---
# Adapted to render in Streamlit via st.pyplot()

def plot_yield_results(d_meas, d_unc, t_meas, t_unc, W_nom, W_unc_range, rho, a, shock_model, burst_type, pdf_pages):
    fig = plt.figure(figsize=(9.0, 6.5))
    plt.grid(True, which='both', linestyle=':')

    W_best_plot = W_nom
    W_min_plot, W_max_plot = W_unc_range

    # generate model curve by calculating yield forward from range of distances
    d_curve = np.linspace(0.95*d_meas, 1.05*d_meas, 200)
    w_curve = []
    w_curve_low = []   # yield if time was slower (t + t_unc)
    w_curve_high = []  # yield if time was faster (t - t_unc)

    for d in d_curve:
        w_curve.append(find_yield_for_measurement(d, t_meas, rho, a, shock_model))
        w_curve_low.append(find_yield_for_measurement(d, t_meas + t_unc, rho, a, shock_model))
        w_curve_high.append(find_yield_for_measurement(d, t_meas - t_unc, rho, a, shock_model))

    w_curve = np.array(w_curve)
    w_curve_low = np.array(w_curve_low)
    w_curve_high = np.array(w_curve_high)

    valid_indices = ~np.isnan(w_curve) & ~np.isnan(w_curve_low) & ~np.isnan(w_curve_high)
    d_curve = d_curve[valid_indices]
    w_curve = w_curve[valid_indices]
    w_curve_low = w_curve_low[valid_indices]
    w_curve_high = w_curve_high[valid_indices]

    if burst_type == 'surface':
        w_curve = 0.5 * w_curve
        w_curve_low = 0.5 * w_curve_low
        w_curve_high = 0.5 * w_curve_high
        W_best_plot *= 0.5
        W_min_plot  *= 0.5
        W_max_plot  *= 0.5
    
    plt.plot(w_curve, d_curve, color='blue', label='Model Relationship')
    plt.fill_betweenx(d_curve, w_curve_low, w_curve_high, color='blue', alpha=0.15, label='Timing Uncertainty Envelope')
    plt.axhspan(d_meas - d_unc, d_meas + d_unc, color='green', alpha=0.1, label=f'Distance Uncertainty')
    plt.axhline(d_meas, linestyle='-', color='green', label=f'Measured Distance = {d_meas:.1f} m')
    plt.axhline(d_meas + d_unc, linestyle='--', color='green', alpha=0.7)
    plt.axhline(d_meas - d_unc, linestyle='--', color='green', alpha=0.7)
    plt.axvline(W_best_plot, linestyle='--', color='red', label=f'Best Yield = {W_best_plot:.3f} kt [{W_min_plot:.3f} to {W_max_plot:.3f}] kt')
    plt.axvspan(W_min_plot, W_max_plot, color='red', alpha=0.2, label=f'Yield Uncertainty')
    plt.axvline(W_min_plot, linestyle=':', color='red')
    plt.axvline(W_max_plot, linestyle=':', color='red')
    
    plt.xlabel('Yield (kt)')
    plt.ylabel('Distance (m)')
    title = f'Yield vs. Distance ({shock_model.capitalize()} Model, {burst_type.title()} Burst)'
    plt.title(title)
    plt.xscale('log')
    plt.yscale('log')
    if len(w_curve) > 0:
        plt.xlim(min(w_curve), max(w_curve))
    plt.legend(loc='best')
    
    plt.gcf().text(.06, .95, 'UNCLASSIFIED', fontsize=10, color='green')
    plt.gcf().text(.78, .05, 'UNCLASSIFIED', fontsize=10, color='green')

    pdf_pages.savefig(fig)
    st.pyplot(fig, use_container_width=False) 
    plt.close(fig)

# function to plot the original data and the fitted curve
def plot_fit(W_best, rho, a, d_meas, t_meas, u, v, w, p, q):

    # convert yield in kt to joules
    W_best_joules = W_best * ktj

    # define characteristic length and time scales
    lc = (W_best_joules / (rho * a**2.))**(1. / 3.)  # characteristic length scale
    tc = lc / a                                      # characteristic time scale

    # define log time-sampled region to fit functions to
    t_fit = np.logspace(-4, 3, num=1000, base=10)
    tstar_fit = t_fit / tc

    # plot strong shock regime (Taylor solution)
    rstar_strong = tstar_fit**(2. / 5.)

    # plot weak shock regime (Wei-Hargather solution)
    rstar_weak = tstar_fit + u * (np.log(v + w * (tstar_fit)**p))**q

    # plot acoustic regime (sound speed)
    rstar_acoustic = tstar_fit

    fig = plt.figure(figsize=(9.0, 6.5))
    plt.grid(True)
    plt.scatter(tstar_fit, rstar_strong, s=6, color='red', label='Strong Shock Solution')
    plt.scatter(tstar_fit, rstar_weak, s=6, color='orange', label='Weak Shock Solution')
    plt.scatter(tstar_fit, rstar_acoustic, s=6, color='green', label='Acoustic Solution')

    # scale measured distance and time and plot
    d_scaled = d_meas / lc
    t_scaled = t_meas / tc

    plt.scatter(t_scaled, d_scaled, s=24, color='blue', label='Scaled Measured Distance & Time')

    plt.xlabel('Scaled Time')
    plt.ylabel('Scaled Radius')
    plt.title('Strong, Weak, and Acoustic Radius vs Time Plots', fontsize=12, color='black')
    plt.xscale('log')
    plt.yscale('log')

    plt.gcf().text(.06, .95, 'UNCLASSIFIED', fontsize=10, color='green')
    plt.gcf().text(.78, .05, 'UNCLASSIFIED', fontsize=10, color='green')

    plt.legend(loc='upper left')
    st.pyplot(fig, use_container_width=False) 
    pdf_pages.savefig(fig)

def plot_hob_vs_yield(W_guesses, hob_calc_array, hob_dp_array, hob_dm_array, hob_best, hob_min, hob_max, t_meas, W_nom, W_low, W_high, shock_model, pdf_pages):
    fig = plt.figure(figsize=(9.0, 6.5))
    plt.grid(True, which='both', linestyle=':')
    
    valid_indices = ~np.isnan(hob_calc_array)
    W_guesses = W_guesses[valid_indices]
    hob_calc_array = hob_calc_array[valid_indices]

    hob_upper = hob_calc_array + hob_dp_array[valid_indices]
    hob_lower = hob_calc_array - hob_dm_array[valid_indices]

    plt.plot(W_guesses, hob_calc_array, color='blue', label=f'Model Relationship (t = {t_meas:.3f} s)')
    plt.fill_between(W_guesses, hob_lower, hob_upper, color='blue', alpha=0.15, label='Timing Uncertainty Envelope')

    plt.axvline(W_nom, linestyle='-', color='red', label=f'Nominal Yield = {W_nom:.3f} kt')
    plt.axvline(W_low, linestyle='--', color='red')
    plt.axvline(W_high, linestyle='--', color='red')
    plt.axvspan(W_low, W_high, color='red', alpha=0.1, label=f'Yield Uncertainty')
    plt.axhline(hob_best, linestyle='--', color='green', label=f'Best HOB (AGL) = {hob_best:.1f} m [{hob_min:.1f} to {hob_max:.1f}] m')
    
    if len(W_guesses) > 0:
        plt.fill_betweenx([hob_min, hob_max], W_guesses.min(), W_guesses.max(), color='green', alpha=0.2, label='HOB Uncertainty')
        plt.xlim(W_guesses.min(), W_guesses.max())
        
    plt.axhline(hob_min, linestyle=':', color='green')
    plt.axhline(hob_max, linestyle=':', color='green')

    plt.xlabel('Yield (kt)')
    plt.ylabel('Height of Burst AGL (m)')
    plt.title(f'Calculated HOB vs. Yield ({shock_model.capitalize()} Shock Model)')
    plt.legend(loc='best')
    
    plt.gcf().text(.06, .95, 'UNCLASSIFIED', fontsize=10, color='green')
    plt.gcf().text(.78, .05, 'UNCLASSIFIED', fontsize=10, color='green')

    pdf_pages.savefig(fig)
    st.pyplot(fig, use_container_width=False) 
    plt.close(fig)

def plot_velocity_profile(hob, W_kt, T_ground_c, shock_model, pdf_pages, terr_alt_m=0.0):
    fig = plt.figure(figsize=(9.0, 6.5))
    plt.grid(True, which='both', linestyle=':')
    
    alts_agl = np.linspace(0, hob, 200)
    velocities = []
    W_j = W_kt * ktj

    for alt_agl in alts_agl:
        r_from_burst = hob - alt_agl
        if r_from_burst < 1e-3:
            velocities.append(np.nan)
            continue
            
        _, _, rho, a = atmos_properties_at_alt(alt_agl + terr_alt_m, T_ground_c, sensor_alt_msl=terr_alt_m)
        lc = (W_j / (rho * a**2))**(1.0/3.0)
        rstar_target = r_from_burst / lc
        try:
            if shock_model == 'strong':
                tstar_sol = rstar_target**(2.5)
                U_star = 0.4 * (tstar_sol**(-0.6)) if tstar_sol >= 1e-9 else 1.0
            else:
                def rstar_residual(ts):
                    if ts <= 0: return rstar_target
                    return ts + u * (np.log(v + w * ts**p))**q - rstar_target
                tstar_sol = brentq(rstar_residual, 1e-9, rstar_target + 1.0)
                log_val = np.log(v + w * tstar_sol**p)
                U_star = 1.0 + (u*q*(log_val)**(q-1) * (1.0/(v+w*tstar_sol**p)) * w*p*tstar_sol**(p-1))
            velocities.append(a * U_star)
        except (ValueError, ZeroDivisionError):
            velocities.append(np.nan)
            
    plt.plot(velocities, alts_agl, color='purple')
    
    _, _, _, a_ground = atmos_properties_at_alt(terr_alt_m, T_ground_c, sensor_alt_msl=terr_alt_m)
    plt.axvline(a_ground, color='gray', linestyle='--', label=f'Speed of Sound (ground) = {a_ground:.1f} m/s')

    plt.xlabel('Shock Velocity (m/s)')
    plt.ylabel('Altitude AGL (m)')
    plt.title(f'Shock Velocity Profile for {W_kt:.3f} kt at HOB={hob:.1f} m ({shock_model.capitalize()})')
    plt.legend()
    
    plt.gcf().text(.06, .95, 'UNCLASSIFIED', fontsize=10, color='green')
    plt.gcf().text(.78, .05, 'UNCLASSIFIED', fontsize=10, color='green')
    
    pdf_pages.savefig(fig)
    st.pyplot(fig, use_container_width=False) 
    plt.close(fig)

# --- Streamlit UI Layout ---

st.title("Flash-to-Shock Analysis Tool")
st.markdown("**UNCLASSIFIED** - Calculates Explosive Yield or Height of Burst from flash-to-shock time measurements.")
st.markdown("Author: C.J. Miko, AFTAC/23 ANS/ANA. Current as of: 21 April 2026.")
st.markdown("---")

# Layout columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Analysis Mode")
    mode = st.radio("Select computation:", ["Solve for Yield (Given Distance)", "Solve for HOB (Given Yield)"])
    
    st.subheader("Common Parameters")
    t_meas_str = st.text_input("Measured Time (s):", value="1.75")
    t_unc_str = st.text_input("Time Uncertainty (+/- s):", value="1/30")
    temp_str = st.text_input("Ground Temp (C):", value="25.0")
    model = st.selectbox("Shock Model:", ["weak", "strong"])

with col2:
    st.subheader("Specific Parameters")
    if mode == "Solve for Yield (Given Distance)":
        dist_str = st.text_input("Measured Distance:", value="2380.0")
        dist_unc_str = st.text_input("Distance Uncertainty (+/-):", value="20.0")
        dist_unit = st.selectbox("Distance Unit:", ["ft", "m"])
        alt_str = st.text_input("Burst Altitude MSL (m):", value="0.0")
        burst_type = st.selectbox("Burst Type:", ["free air", "surface"], index=1)
    else: # HOB
        yield_str = st.text_input("Nominal Yield (kt):", value="0.25")
        yield_unc_str = st.text_input("Yield Uncertainty (+/- kt):", value="0.05")
        terr_alt_str = st.text_input("Terrain Elevation MSL (m):", value="0.0")


st.markdown("---")

# --- Execution Logic ---
if st.button("Run Analysis", type="primary"):
    
    log_output = []
    def log(msg):
        log_output.append(msg)
        
    try:
        t_meas = parse_input(t_meas_str)
        t_unc = parse_input(t_unc_str)
        temp_c = parse_input(temp_str)
        
        log(f"--- Starting {mode.split('(')[0].strip()} Analysis ({datetime.datetime.now().strftime('%H:%M:%S')}) ---")
        
        results_container = st.container()
        
        with results_container:
            if mode == "Solve for Yield (Given Distance)":
                d_meas_raw = parse_input(dist_str)
                d_unc_raw = parse_input(dist_unc_str)
                alt_m = parse_input(alt_str)
                
                if dist_unit == 'ft':
                    d_meas = d_meas_raw * ftm
                    d_unc = d_unc_raw * ftm
                else:
                    d_meas, d_unc = d_meas_raw, d_unc_raw

                # For yield calc, the "sensor" is effectively at the burst altitude for property calculation
                _, T_k, rho, a = atmos_properties_at_alt(alt_m, temp_c, sensor_alt_msl=alt_m)
                log(f"Atmosphere at {alt_m}m MSL: T={T_k:.1f} K, rho={rho:.3f} kg/m3, a={a:.1f} m/s")

                w_best = find_yield_for_measurement(d_meas, t_meas, rho, a, model)
                w1 = find_yield_for_measurement(d_meas - d_unc, t_meas + t_unc, rho, a, model)
                w2 = find_yield_for_measurement(d_meas + d_unc, t_meas - t_unc, rho, a, model)
                w3 = find_yield_for_measurement(d_meas - d_unc, t_meas - t_unc, rho, a, model)
                w4 = find_yield_for_measurement(d_meas + d_unc, t_meas + t_unc, rho, a, model)
                
                all_y = [w for w in [w_best, w1, w2, w3, w4] if not np.isnan(w)]
                w_min = min(all_y) if all_y else np.nan
                w_max = max(all_y) if all_y else np.nan

                factor = 0.5 if burst_type == 'surface' else 1.0
                w_best, w_min, w_max = w_best * factor, w_min * factor, w_max * factor
                
                log(f"-> BEST YIELD: {w_best:.3f} kt")
                log(f"-> Uncertainty Bound: {w_min:.3f} to {w_max:.3f} kt")
                
                with open(f"outputs_yield_{model}.owt", "w") as f:
                    f.write(f"Yield Results ({model} model, {burst_type})\n")
                    f.write(f"Best: {w_best:.3f} kt\nRange: {w_min:.3f} - {w_max:.3f} kt\n")
                log(f"File saved: outputs_yield_{model}.owt")
                
                st.code("\n".join(log_output), language="plaintext")
                
                st.subheader("Analysis Visualizations")
                with st.spinner('Generating PDF Plots...'):
                    pdf_pages = PdfPages(f'Yield_Analysis_Plots_{model}.pdf')
                    # Pass the raw free-air-equivalent yields to the plotter
                    plot_yield_results(d_meas, d_unc, t_meas, t_unc, w_best / factor, (w_min / factor, w_max / factor), rho, a, model, burst_type, pdf_pages)
                    plot_fit(w_best, rho, a, d_meas, t_meas, u, v, w, p, q)
                    pdf_pages.close()
                st.success(f"PDF Plot saved locally to: Yield_Analysis_Plots_{model}.pdf")

            else: # Solve for HOB
                w_nom = parse_input(yield_str)
                w_unc = parse_input(yield_unc_str)
                terr_alt_m = parse_input(terr_alt_str)
                
                log(f"Integrating atmospheric path (Base Terrain: {terr_alt_m}m MSL)...")
                h_best = find_hob_for_time(t_meas, w_nom, temp_c, model, terr_alt_m)
                
                w_low = max(0.001, w_nom - w_unc)
                w_high = w_nom + w_unc
                
                h1 = find_hob_for_time(t_meas - t_unc, w_low, temp_c, model, terr_alt_m)
                h2 = find_hob_for_time(t_meas + t_unc, w_low, temp_c, model, terr_alt_m)
                h3 = find_hob_for_time(t_meas - t_unc, w_high, temp_c, model, terr_alt_m)
                h4 = find_hob_for_time(t_meas + t_unc, w_high, temp_c, model, terr_alt_m)

                all_h = [h for h in [h_best, h1, h2, h3, h4] if h is not None and not np.isnan(h)]
                h_min = min(all_h) if all_h else np.nan
                h_max = max(all_h) if all_h else np.nan

                log(f"-> BEST HOB (AGL): {h_best:.1f} m")
                log(f"-> Uncertainty Bound: {h_min:.1f} to {h_max:.1f} m")

                with open(f"outputs_hob_{model}.owt", "w") as f:
                    f.write(f"HOB Results ({model} model, Terrain MSL: {terr_alt_m}m)\n")
                    f.write(f"Best AGL: {h_best:.1f} m\nRange: {h_min:.1f} - {h_max:.1f} m\n")
                log(f"File saved: outputs_hob_{model}.owt")

                st.code("\n".join(log_output), language="plaintext")

                st.subheader("Analysis Visualizations")
                with st.spinner('Generating PDF Plots...'):
                    pdf_pages = PdfPages(f'HOB_Analysis_Plots_{model}.pdf')
                    
                    w_guesses = np.linspace(max(0.01, w_nom - 3*w_unc), w_nom + 3*w_unc, 50)
                    h_calc, h_dp, h_dm = [], [], []
                    for wg in w_guesses:
                        hc = find_hob_for_time(t_meas, wg, temp_c, model, terr_alt_m)
                        hp = find_hob_for_time(t_meas - t_unc, wg, temp_c, model, terr_alt_m)
                        hm = find_hob_for_time(t_meas + t_unc, wg, temp_c, model, terr_alt_m)
                        h_calc.append(hc)
                        h_dp.append(abs(hp - hc) if not np.isnan(hc) and not np.isnan(hp) else 0)
                        h_dm.append(abs(hc - hm) if not np.isnan(hc) and not np.isnan(hm) else 0)
                        
                    plot_hob_vs_yield(np.array(w_guesses), np.array(h_calc), np.array(h_dp), np.array(h_dm), 
                                      h_best, h_min, h_max, t_meas, w_nom, w_low, w_high, model, pdf_pages)
                                      
                    if not np.isnan(h_best):
                        plot_velocity_profile(h_best, w_nom, temp_c, model, pdf_pages, terr_alt_m)
                        
                    pdf_pages.close()
                st.success(f"PDF Plots saved locally to: HOB_Analysis_Plots_{model}.pdf")

    except Exception as e:
        st.error(f"Error during calculation: {e}. Please check your inputs to ensure they are valid numerical values.")
