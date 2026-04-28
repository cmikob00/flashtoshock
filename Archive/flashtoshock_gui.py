# Flash to Shock Analysis GUI
# Adapted by AI, 21 April 2026
# Unified application for determining Yield or Height of Burst.
# UNCLASSIFIED

import math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import brentq
import datetime

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
        raise ValueError

# --- Core Physics & Atmosphere ---

def atmos_properties_at_alt(alt, T_ground_c):
    """Calculates atmospheric properties using a standard lapse rate and Ideal Gas Law."""
    T_ground_k = T_ground_c + 273.15
    L = 0.0065  # Standard tropospheric temperature lapse rate (K/m)
    T_k = max(216.65, T_ground_k - (L * alt))
    P = P0 * math.exp(-alt / H)
    
    # Calculate rho strictly via the Ideal Gas Law
    rho = P / (R_s * T_k)  
    
    a = np.sqrt(gam * R_s * T_k)  
    return P, T_k, rho, a

def get_shock_arrival_time(hob, W_kt, T_ground_c, n_layers=100, shock_model='weak'):
    if hob <= 1.0: return 1e6
    W_j = W_kt * ktj
    dh = hob / n_layers
    total_time = 0.0
    for i in range(n_layers):
        z_mid = (i + 0.5) * dh
        r_from_burst = hob - z_mid
        _, _, rho, a = atmos_properties_at_alt(z_mid, T_ground_c)
        
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

def find_hob_for_time(target_time, W_kt, T_ground_c, shock_model='weak'):
    def time_residual(hob):
        return get_shock_arrival_time(hob, W_kt, T_ground_c, shock_model=shock_model) - target_time
    try:
        return brentq(time_residual, 1.1, 50000.0) 
    except ValueError:
        return np.nan

def find_yield_for_measurement(d_meas, t_meas, rho, a, shock_model='weak'):
    if t_meas <= 0 or d_meas <= 0: return np.nan
    if shock_model == 'strong':
        try:
            # Dimensionless ratio r* / t* = d / (a*t) calculated from rearranging the lc and tc equations
            ratio = d_meas / (a * t_meas)
            tstar = ratio**(-5.0 / 3.0)
        except (ValueError, ZeroDivisionError):
            return np.nan
    else: 
        def residual(tstar):
            if tstar <= 0: return 1e6
            model_rstar = tstar + u * (np.log(v + w * tstar**p))**q
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
    plt.close(fig)

def plot_hob_vs_yield(W_guesses, hob_calc_array, hob_dp_array, hob_dm_array, hob_best, hob_min, hob_max, t_meas, W_nom, W_low, W_high, shock_model, pdf_pages):

    fig = plt.figure(figsize=(9.0, 6.5))
    plt.grid(True, which='both', linestyle=':')
    
    # filter out NaNs
    valid_indices = ~np.isnan(hob_calc_array)
    W_guesses = W_guesses[valid_indices]
    hob_calc_array = hob_calc_array[valid_indices]

    # recover absolute upper and lower HOB bounds from the delta arrays
    hob_upper = hob_calc_array + hob_dp_array[valid_indices]
    hob_lower = hob_calc_array - hob_dm_array[valid_indices]

    # plot the smooth central line and shaded Timing Uncertainty Envelope
    plt.plot(W_guesses, hob_calc_array, color='blue', label=f'Model Relationship (t = {t_meas:.3f} s)')
    plt.fill_between(W_guesses, hob_lower, hob_upper, color='blue', alpha=0.15, label='Timing Uncertainty Envelope')

    plt.axvline(W_nom, linestyle='-', color='red', label=f'Nominal Yield = {W_nom:.3f} kt')
    plt.axvline(W_low, linestyle='--', color='red')
    plt.axvline(W_high, linestyle='--', color='red')
    plt.axvspan(W_low, W_high, color='red', alpha=0.1, label=f'Yield Uncertainty')
    plt.axhline(hob_best, linestyle='--', color='green', label=f'Best HOB = {hob_best:.1f} m [{hob_min:.1f} to {hob_max:.1f}] m')
    
    if len(W_guesses) > 0:
        plt.fill_betweenx([hob_min, hob_max], W_guesses.min(), W_guesses.max(), color='green', alpha=0.2, label='HOB Uncertainty')
        plt.xlim(W_guesses.min(), W_guesses.max())
        
    plt.axhline(hob_min, linestyle=':', color='green')
    plt.axhline(hob_max, linestyle=':', color='green')

    plt.xlabel('Yield (kt)')
    plt.ylabel('Height of Burst (m)')
    plt.title(f'Calculated HOB vs. Yield ({shock_model.capitalize()} Shock Model)')
    plt.legend(loc='best')
    
    plt.gcf().text(.06, .95, 'UNCLASSIFIED', fontsize=10, color='green')
    plt.gcf().text(.78, .05, 'UNCLASSIFIED', fontsize=10, color='green')
    pdf_pages.savefig(fig)
    plt.close(fig)

def plot_velocity_profile(hob, W_kt, T_ground_c, shock_model, pdf_pages):
    fig = plt.figure(figsize=(9.0, 6.5))
    plt.grid(True, which='both', linestyle=':')
    
    alts = np.linspace(0, hob, 200)
    velocities = []
    W_j = W_kt * ktj

    for alt in alts:
        r_from_burst = hob - alt
        if r_from_burst < 1e-3:
            velocities.append(np.nan)
            continue
        _, _, rho, a = atmos_properties_at_alt(alt, T_ground_c)
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
            
    plt.plot(velocities, alts, color='purple')
    _, _, _, a_ground = atmos_properties_at_alt(0, T_ground_c)
    plt.axvline(a_ground, color='gray', linestyle='--', label=f'Speed of Sound (ground) = {a_ground:.1f} m/s')

    plt.xlabel('Shock Velocity (m/s)')
    plt.ylabel('Altitude (m)')
    plt.title(f'Shock Velocity Profile for {W_kt:.3f} kt at HOB={hob:.1f} m ({shock_model.capitalize()})')
    plt.legend()
    
    plt.gcf().text(.06, .95, 'UNCLASSIFIED', fontsize=10, color='green')
    plt.gcf().text(.78, .05, 'UNCLASSIFIED', fontsize=10, color='green')
    pdf_pages.savefig(fig)
    plt.close(fig)

# --- GUI Application ---

class FlashToShockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flash-to-Shock Analysis Tool (UNCLASSIFIED)")
        self.root.geometry("600x750")
        
        self.mode = tk.StringVar(value="Yield") # Default mode
        
        self.create_widgets()
        self.update_input_fields() # Initial setup

    def create_widgets(self):
        # 1. Mode Selection Frame
        mode_frame = ttk.LabelFrame(self.root, text="Analysis Mode")
        mode_frame.pack(fill="x", padx=10, pady=5)
        ttk.Radiobutton(mode_frame, text="Solve for Yield (Given Distance)", variable=self.mode, value="Yield", command=self.update_input_fields).pack(side="left", padx=10, pady=5)
        ttk.Radiobutton(mode_frame, text="Solve for HOB (Given Yield)", variable=self.mode, value="HOB", command=self.update_input_fields).pack(side="left", padx=10, pady=5)

        # 2. Common Inputs Frame
        common_frame = ttk.LabelFrame(self.root, text="Common Parameters")
        common_frame.pack(fill="x", padx=10, pady=5)
        
        self.t_meas_var = tk.StringVar(value="1.75")
        self.t_unc_var = tk.StringVar(value=str(round(1/30, 4)))
        self.temp_var = tk.StringVar(value="25.0")
        self.model_var = tk.StringVar(value="weak")

        ttk.Label(common_frame, text="Time (s):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(common_frame, textvariable=self.t_meas_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(common_frame, text="+/-").grid(row=0, column=2)
        ttk.Entry(common_frame, textvariable=self.t_unc_var, width=10).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(common_frame, text="Ground Temp (C):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(common_frame, textvariable=self.temp_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(common_frame, text="Shock Model:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(common_frame, textvariable=self.model_var, values=["weak", "strong"], state="readonly", width=8).grid(row=2, column=1, padx=5, pady=2)

        # 3. Dynamic Inputs Frame
        self.dynamic_frame = ttk.LabelFrame(self.root, text="Specific Parameters")
        self.dynamic_frame.pack(fill="x", padx=10, pady=5)
        
        # Variables for Yield mode
        self.dist_var = tk.StringVar(value="2380.0")
        self.dist_unc_var = tk.StringVar(value="20.0")
        self.dist_unit_var = tk.StringVar(value="ft")
        self.alt_var = tk.StringVar(value="0.0")
        self.burst_type_var = tk.StringVar(value="surface")
        
        # Variables for HOB mode
        self.yield_var = tk.StringVar(value="0.25")
        self.yield_unc_var = tk.StringVar(value="0.05")

        # 4. Action Frame
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(action_frame, text="Run Analysis", command=self.run_analysis, style="Accent.TButton").pack(side="left", padx=5)
        ttk.Button(action_frame, text="Clear Output", command=lambda: self.output_text.delete(1.0, tk.END)).pack(side="left", padx=5)
        ttk.Button(action_frame, text="Exit", command=self.root.quit).pack(side="right", padx=5)

        # 5. Output Frame
        output_frame = ttk.LabelFrame(self.root, text="Console Output")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Courier", 9))
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

    def update_input_fields(self):
        # Clear dynamic frame
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        if self.mode.get() == "Yield":
            ttk.Label(self.dynamic_frame, text="Distance:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(self.dynamic_frame, textvariable=self.dist_var, width=10).grid(row=0, column=1, padx=5, pady=2)
            ttk.Label(self.dynamic_frame, text="+/-").grid(row=0, column=2)
            ttk.Entry(self.dynamic_frame, textvariable=self.dist_unc_var, width=10).grid(row=0, column=3, padx=5, pady=2)
            ttk.Combobox(self.dynamic_frame, textvariable=self.dist_unit_var, values=["ft", "m"], state="readonly", width=5).grid(row=0, column=4, padx=5, pady=2)

            ttk.Label(self.dynamic_frame, text="Burst Altitude MSL (m):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(self.dynamic_frame, textvariable=self.alt_var, width=10).grid(row=1, column=1, padx=5, pady=2)

            ttk.Label(self.dynamic_frame, text="Burst Type:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
            ttk.Combobox(self.dynamic_frame, textvariable=self.burst_type_var, values=["free_air", "surface"], state="readonly", width=10).grid(row=2, column=1, padx=5, pady=2)

        else: # HOB
            ttk.Label(self.dynamic_frame, text="Nominal Yield (kt):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(self.dynamic_frame, textvariable=self.yield_var, width=10).grid(row=0, column=1, padx=5, pady=2)
            ttk.Label(self.dynamic_frame, text="+/-").grid(row=0, column=2)
            ttk.Entry(self.dynamic_frame, textvariable=self.yield_unc_var, width=10).grid(row=0, column=3, padx=5, pady=2)

    def log(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update()

    def run_analysis(self):
        try:
            t_meas = parse_input(self.t_meas_var.get())
            t_unc = parse_input(self.t_unc_var.get())
            temp_c = parse_input(self.temp_var.get())
            model = self.model_var.get()
            
            self.log(f"\n--- Starting {self.mode.get()} Analysis ({datetime.datetime.now().strftime('%H:%M:%S')}) ---")
            
            if self.mode.get() == "Yield":
                self.solve_yield(t_meas, t_unc, temp_c, model)
            else:
                self.solve_hob(t_meas, t_unc, temp_c, model)
                
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all numeric fields contain valid numbers or valid simple fractions (e.g. '1/30').")

    def solve_yield(self, t_meas, t_unc, temp_c, model):
        d_meas_raw = parse_input(self.dist_var.get())
        d_unc_raw = parse_input(self.dist_unc_var.get())
        alt_m = parse_input(self.alt_var.get())
        b_type = self.burst_type_var.get()
        
        # Unit conversion
        if self.dist_unit_var.get() == 'ft':
            d_meas = d_meas_raw * ftm
            d_unc = d_unc_raw * ftm
        else:
            d_meas, d_unc = d_meas_raw, d_unc_raw

        _, T_k, rho, a = atmos_properties_at_alt(alt_m, temp_c)
        self.log(f"Atmosphere: T={T_k:.1f} K, rho={rho:.3f} kg/m3, a={a:.1f} m/s")

        w_best = find_yield_for_measurement(d_meas, t_meas, rho, a, model)
        w1     = find_yield_for_measurement(d_meas - d_unc, t_meas + t_unc, rho, a, model)
        w2     = find_yield_for_measurement(d_meas + d_unc, t_meas - t_unc, rho, a, model)
        w3     = find_yield_for_measurement(d_meas - d_unc, t_meas - t_unc, rho, a, model)
        w4     = find_yield_for_measurement(d_meas + d_unc, t_meas + t_unc, rho, a, model)
        
        all_y = [w for w in [w_best, w1, w2, w3, w4] if not np.isnan(w)]
        w_min = min(all_y) if all_y else np.nan
        w_max = max(all_y) if all_y else np.nan

        factor = 0.5 if b_type == 'surface' else 1.0
        w_best, w_min, w_max = w_best * factor, w_min * factor, w_max * factor
        
        self.log(f"-> BEST YIELD: {w_best:.3f} kt")
        self.log(f"-> Uncertainty Bound: {w_min:.3f} to {w_max:.3f} kt")
        
        # Save text output dynamically
        with open(f"outputs_yield_{model}.owt", "w") as f:
            f.write(f"Yield Results ({model} model, {b_type})\n")
            f.write(f"Best: {w_best:.3f} kt\nRange: {w_min:.3f} - {w_max:.3f} kt\n")
        self.log(f"File saved: outputs_yield_{model}.owt")
        
        # Generate Plot
        self.log("Generating PDF Plot...")
        pdf_pages = PdfPages(f'Yield_Analysis_Plots_{model}.pdf')
        plot_yield_results(d_meas, d_unc, t_meas, t_unc, w_best / factor, (w_min / factor, w_max / factor), rho, a, model, b_type, pdf_pages)
        pdf_pages.close()
        self.log(f"File saved: Yield_Analysis_Plots_{model}.pdf")
        
    def solve_hob(self, t_meas, t_unc, temp_c, model):
        w_nom = parse_input(self.yield_var.get())
        w_unc = parse_input(self.yield_unc_var.get())
        
        self.log("Integrating atmospheric path...")
        h_best = find_hob_for_time(t_meas, w_nom, temp_c, model)
        
        # Prevent negative yield inputs on the lower bounds
        w_low = max(0.001, w_nom - w_unc)
        w_high = w_nom + w_unc
        
        try: h1    = find_hob_for_time(t_meas - t_unc, w_low, temp_c, model)
        except: h1 = np.nan
        try: h2    = find_hob_for_time(t_meas + t_unc, w_low, temp_c, model)
        except: h2 = np.nan
        try: h3    = find_hob_for_time(t_meas - t_unc, w_high, temp_c, model)
        except: h3 = np.nan
        try: h4    = find_hob_for_time(t_meas + t_unc, w_high, temp_c, model)
        except: h4 = np.nan
        
        all_h = [h for h in [h_best, h1, h2, h3, h4] if not np.isnan(h)]
        h_min = min(all_h) if all_h else np.nan
        h_max = max(all_h) if all_h else np.nan
        
        self.log(f"-> BEST HOB: {h_best:.1f} m")
        self.log(f"-> Uncertainty Bound: {h_min:.1f} to {h_max:.1f} m")
        
        # Save text output dynamically
        with open(f"outputs_hob_{model}.owt", "w") as f:
            f.write(f"HOB Results ({model} model)\n")
            f.write(f"Best: {h_best:.1f} m\nRange: {h_min:.1f} - {h_max:.1f} m\n")
        self.log(f"File saved: outputs_hob_{model}.owt")
        
        # Generate Plots
        self.log("Generating PDF Plots...")
        pdf_pages = PdfPages(f'HOB_Analysis_Plots_{model}.pdf')
        
        # Recreate Monte Carlo arrays for plotting
        w_guesses = np.linspace(max(0.01, w_nom - 4*w_unc), w_nom + 4*w_unc, 50)
        h_calc, h_dp, h_dm = [], [], []
        for wg in w_guesses:
            hc = find_hob_for_time(t_meas, wg, temp_c, model)
            hp = find_hob_for_time(t_meas - t_unc, wg, temp_c, model)
            hm = find_hob_for_time(t_meas + t_unc, wg, temp_c, model)
            h_calc.append(hc)
            h_dp.append(abs(hp - hc) if not np.isnan(hc) and not np.isnan(hp) else 0)
            h_dm.append(abs(hc - hm) if not np.isnan(hc) and not np.isnan(hm) else 0)
            
        plot_hob_vs_yield(np.array(w_guesses), np.array(h_calc), np.array(h_dp), np.array(h_dm), 
                          h_best, h_min, h_max, t_meas, w_nom, w_low, w_high, model, pdf_pages)
                          
        if not np.isnan(h_best):
            plot_velocity_profile(h_best, w_nom, temp_c, model, pdf_pages)
            
        pdf_pages.close()
        self.log(f"File saved: HOB_Analysis_Plots_{model}.pdf")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Use standard modern styling
    try:
        style = ttk.Style()
        style.theme_use('clam')
    except tk.TclError:
        pass # Fallback to default if 'clam' is missing
    
    app = FlashToShockApp(root)
    root.mainloop()
