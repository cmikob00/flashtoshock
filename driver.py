'''
Streamlit Driver for Flash to Shock
'''

#!/usr/bin/env python3
from pathlib import Path
import sys

project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from flashtoshock.constants import *
from flashtoshock.parsers import *
from flashtoshock.atmos import *
from flashtoshock.hob_solver import *
from flashtoshock.yield_solver import *
from flashtoshock.plotters import *

if __name__ == "__main__":

    # --- Streamlit UI Layout ---
    st.title("Flash-to-Shock Analysis Tool")
    st.markdown("**UNCLASSIFIED** - Calculates Explosive Yield or Height of Burst from flash-to-shock time measurements.")
    st.markdown("Uses Wei & Hargather (2021) Blast Wave Profile.")
    st.markdown("Author: C.J. Miko, AFTAC/23 ANS/ANA. Current as of: 30 May 2026.")
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
                        plot_fit(w_best, rho, a, d_meas, t_meas, u, v, w, p, q, pdf_pages)
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
                                        h_best, h_min, h_max, t_meas, w_nom, w_low, w_high, model, pdf_pages,
                                        terr_alt_m)
                                        
                        if not np.isnan(h_best):
                            plot_velocity_profile(h_best, w_nom, temp_c, model, pdf_pages, terr_alt_m)
                            
                        pdf_pages.close()
                    st.success(f"PDF Plots saved locally to: HOB_Analysis_Plots_{model}.pdf")

        except Exception as e:
            st.error(f"Error during calculation: {e}. Please check your inputs to ensure they are valid numerical values.")