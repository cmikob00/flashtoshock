'''
Plotting Functions for Flash to Shock
'''

from flashtoshock.constants import *
from flashtoshock.hob_solver import *
from flashtoshock.yield_solver import *

# --- Plotting Functions ---
# Adapted to render in Streamlit via st.pyplot()

def plot_yield_results(
    d_meas,
    d_unc,
    t_meas,
    t_unc,
    W_nom,
    W_unc_range,
    rho,
    a,
    shock_model,
    burst_type,
    pdf_pages
):

    fig = plt.figure(figsize=(9.0, 6.5))
    plt.grid(True, which='both', linestyle=':')

    W_best_plot = W_nom
    W_min_plot, W_max_plot = W_unc_range

    # ---------------------------------------------------------
    # Plotting range controls
    # ---------------------------------------------------------

    # d_fudge_low  = 0.9
    # d_fudge_high = 1.1

    # # logarithmically spaced distance array
    # d_curve = np.logspace(
    #     np.log10(max(1.0, d_meas * d_fudge_low)),
    #     np.log10(d_meas * d_fudge_high),
    #     500
    # )

    d_curve = np.linspace(0.5*d_meas, 2.0*d_meas, 300)

    # ---------------------------------------------------------
    # Generate yield curves
    # ---------------------------------------------------------

    w_curve      = []
    w_curve_low  = []
    w_curve_high = []

    for d in d_curve:

        # nominal timing
        w_curve.append(
            find_yield_for_measurement(
                d,
                t_meas,
                rho,
                a,
                shock_model
            )
        )

        # slower arrival -> lower yield
        w_curve_low.append(
            find_yield_for_measurement(
                d,
                t_meas + t_unc,
                rho,
                a,
                shock_model
            )
        )

        # faster arrival -> higher yield
        w_curve_high.append(
            find_yield_for_measurement(
                d,
                t_meas - t_unc,
                rho,
                a,
                shock_model
            )
        )

    w_curve      = np.array(w_curve)
    w_curve_low  = np.array(w_curve_low)
    w_curve_high = np.array(w_curve_high)

    # ---------------------------------------------------------
    # Remove invalid values
    # ---------------------------------------------------------

    valid_indices = (
        np.isfinite(w_curve) &
        np.isfinite(w_curve_low) &
        np.isfinite(w_curve_high)
    )

    d_curve      = d_curve[valid_indices]
    w_curve      = w_curve[valid_indices]
    w_curve_low  = w_curve_low[valid_indices]
    w_curve_high = w_curve_high[valid_indices]

    # ---------------------------------------------------------
    # Surface burst correction
    # ---------------------------------------------------------

    if burst_type == 'surface':

        w_curve      *= 0.5
        w_curve_low  *= 0.5
        w_curve_high *= 0.5

        W_best_plot  *= 0.5
        W_min_plot   *= 0.5
        W_max_plot   *= 0.5

    # ---------------------------------------------------------
    # Plot model relationship
    # ---------------------------------------------------------

    plt.plot(
        w_curve,
        d_curve,
        color='blue',
        linewidth=2,
        label='Model Relationship'
    )

    plt.fill_betweenx(
        d_curve,
        w_curve_low,
        w_curve_high,
        color='blue',
        alpha=0.15,
        label='Timing Uncertainty Envelope'
    )

    # ---------------------------------------------------------
    # Measured distance and uncertainty
    # ---------------------------------------------------------

    plt.axhspan(
        d_meas - d_unc,
        d_meas + d_unc,
        color='green',
        alpha=0.10,
        label='Distance Uncertainty'
    )

    plt.axhline(
        d_meas,
        linestyle='--',
        color='green',
        linewidth=1.5,
        label=f'Measured Distance = {d_meas:.1f} m'
    )

    plt.axhline(
        d_meas + d_unc,
        linestyle=':',
        color='green',
        alpha=0.7
    )

    plt.axhline(
        d_meas - d_unc,
        linestyle=':',
        color='green',
        alpha=0.7
    )

    # ---------------------------------------------------------
    # Yield estimate and uncertainty
    # ---------------------------------------------------------

    plt.axvline(
        W_best_plot,
        linestyle='-',
        color='red',
        linewidth=1.5,
        label=(
            f'Best Yield = {W_best_plot:.3f} kt '
            f'[{W_min_plot:.3f} to {W_max_plot:.3f}] kt'
        )
    )

    plt.axvspan(
        W_min_plot,
        W_max_plot,
        color='red',
        alpha=0.20,
        label='Yield Uncertainty'
    )

    plt.axvline(
        W_min_plot,
        linestyle='--',
        color='red'
    )

    plt.axvline(
        W_max_plot,
        linestyle='--',
        color='red'
    )

    # ---------------------------------------------------------
    # Axis scaling
    # ---------------------------------------------------------

    plt.xscale('log')
    plt.yscale('log')

    x_min = 0.5 * W_min_plot
    x_max = 2.0 * W_max_plot
    plt.xlim(x_min, x_max)

    d_min = 0.95 * (d_meas - d_unc)
    d_max = 1.05 * (d_meas + d_unc)
    plt.ylim(d_min, d_max)

    # ---------------------------------------------------------
    # Labels and title
    # ---------------------------------------------------------

    plt.xlabel('Yield (kt)')
    plt.ylabel('Distance (m)')

    title = (
        f'Yield vs. Distance '
        f'({shock_model.capitalize()} Model, '
        f'{burst_type.title()} Burst)'
    )

    plt.title(title)

    plt.legend(loc='best')

    # ---------------------------------------------------------
    # Classification markings
    # ---------------------------------------------------------

    plt.gcf().text(
        .06,
        .95,
        'UNCLASSIFIED',
        fontsize=10,
        color='green'
    )

    plt.gcf().text(
        .78,
        .05,
        'UNCLASSIFIED',
        fontsize=10,
        color='green'
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    pdf_pages.savefig(fig)

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)


# function to plot the original data and the fitted curve
def plot_fit(W_best, rho, a, d_meas, t_meas, u, v, w, p, q, pdf_pages):

    fig = plt.figure(figsize=(9.0, 6.5))

    # convert yield in kt to joules
    W_best_joules = W_best * ktj

    # define characteristic length and time scales
    lc = (W_best_joules / (rho * a**2.))**(1. / 3.)  # characteristic length scale
    tc = lc / a                                      # characteristic time scale

    # define log time-sampled region to fit functions to
    t_fit = np.logspace(-4, 4, num=1000, base=10)
    tstar_fit = t_fit / tc

    # plot strong shock regime (Taylor solution)
    rstar_strong = tstar_fit**(2. / 5.)

    # plot weak shock regime (Wei-Hargather solution)
    rstar_weak = tstar_fit + u * (np.log(v + w * (tstar_fit)**p))**q

    # plot acoustic regime (sound speed)
    rstar_acoustic = tstar_fit

    plt.minorticks_on()

    plt.grid(
        True,
        which='major',
        linestyle='-',
        alpha=0.35
    )

    plt.grid(
        True,
        which='minor',
        linestyle=':',
        alpha=0.20
    )

    plt.plot(tstar_fit, rstar_strong,
        color='red',
        linewidth=2,
        label='Strong Shock Solution'
        )
    
    plt.plot(tstar_fit, rstar_weak,
        color='orange',
        linewidth=2,
        label='Weak Shock Solution'
        )
    
    plt.plot(tstar_fit, rstar_acoustic,
        color='green',
        linewidth=2,
        label='Acoustic Solution'
        )

    # scale measured distance and time and plot
    d_scaled = d_meas / lc
    t_scaled = t_meas / tc

    # t_scaled_unc = t_unc / tc
    # d_scaled_unc = d_unc / lc

    plt.errorbar(
        t_scaled,
        d_scaled,
        # xerr=t_scaled_unc,
        # yerr=d_scaled_unc,
        xerr=0,
        yerr=0,
        fmt='o',
        color='blue',
        capsize=3,
        label='Scaled Measurement'
    )

    plt.text(
        0.85,
        0.04,
        f'$l_c$ = {lc:.1f} m\n$t_c$ = {tc:.3f} s',
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8)
    )

    plt.xlabel('Scaled Time')
    plt.ylabel('Scaled Radius')
    plt.title('Strong, Weak, and Acoustic Radius vs Time Plots', fontsize=12, color='black')
    plt.xscale('log')
    plt.yscale('log')

    plt.gcf().text(.06, .95, 'UNCLASSIFIED', fontsize=10, color='green')
    plt.gcf().text(.78, .05, 'UNCLASSIFIED', fontsize=10, color='green')

    plt.legend(loc='upper left')

    pdf_pages.savefig(fig)
    st.pyplot(fig, use_container_width=False) 
    plt.close(fig)


def plot_hob_vs_yield(
    W_guesses,
    hob_calc_array,
    hob_dp_array,
    hob_dm_array,
    hob_best,
    hob_min,
    hob_max,
    t_meas,
    W_nom,
    W_low,
    W_high,
    shock_model,
    pdf_pages,
    terr_alt_m
):

    fig = plt.figure(figsize=(9.0, 6.5))

    # ---------------------------------------------------------
    # Grid styling
    # ---------------------------------------------------------

    plt.minorticks_on()

    plt.grid(
        True,
        which='major',
        linestyle='-',
        alpha=0.35
    )

    plt.grid(
        True,
        which='minor',
        linestyle=':',
        alpha=0.20
    )

    # ---------------------------------------------------------
    # Remove invalid values
    # ---------------------------------------------------------

    valid_indices = np.isfinite(hob_calc_array)

    W_guesses      = W_guesses[valid_indices]
    hob_calc_array = hob_calc_array[valid_indices]
    hob_dp_array   = hob_dp_array[valid_indices]
    hob_dm_array   = hob_dm_array[valid_indices]

    if len(W_guesses) == 0:
        st.warning("No valid HOB values available for plotting.")
        return

    # ---------------------------------------------------------
    # Construct uncertainty envelope safely
    # ---------------------------------------------------------

    hob_upper = np.maximum(
        hob_calc_array + hob_dp_array,
        hob_calc_array
    )

    hob_lower = np.maximum(
        0.0,
        np.minimum(
            hob_calc_array - hob_dm_array,
            hob_calc_array
        )
    )

    # ---------------------------------------------------------
    # Plot main HOB relationship
    # ---------------------------------------------------------

    plt.plot(
        W_guesses,
        hob_calc_array,
        color='blue',
        linewidth=2,
        label=f'Model Relationship (t = {t_meas:.3f} s)'
    )

    plt.fill_between(
        W_guesses,
        hob_lower,
        hob_upper,
        color='blue',
        alpha=0.15,
        label='Timing Uncertainty Envelope'
    )

    # ---------------------------------------------------------
    # Yield uncertainty region
    # ---------------------------------------------------------

    plt.axvline(
        W_nom,
        linestyle='--',
        color='red',
        linewidth=1.5,
        label=f'Nominal Yield = {W_nom:.3f} kt'
    )

    plt.axvline(
        W_low,
        linestyle=':',
        color='red'
    )

    plt.axvline(
        W_high,
        linestyle=':',
        color='red'
    )

    plt.axvspan(
        W_low,
        W_high,
        color='red',
        alpha=0.10,
        label='Yield Uncertainty'
    )

    # ---------------------------------------------------------
    # HOB uncertainty region
    # ---------------------------------------------------------

    plt.axhline(
        hob_best,
        linestyle='-',
        color='green',
        linewidth=1.5,
        label=(
            f'Best HOB (AGL) = {hob_best:.1f} m '
            f'[{hob_min:.1f} to {hob_max:.1f}] m'
        )
    )

    plt.axhline(
        hob_min,
        linestyle='--',
        color='green'
    )

    plt.axhline(
        hob_max,
        linestyle='--',
        color='green'
    )

    plt.fill_betweenx(
        [hob_min, hob_max],
        W_guesses.min(),
        W_guesses.max(),
        color='green',
        alpha=0.08,
        label='HOB Uncertainty'
    )

    # ---------------------------------------------------------
    # Axis scaling
    # ---------------------------------------------------------

    plt.xscale('log')
    plt.yscale('log')

    x_min = 0.8 * W_low
    x_max = 1.2 * W_high
    plt.xlim(x_min, x_max)

    y_min = 0.8 * hob_min
    y_max = 1.2 * hob_max
    plt.ylim(y_min, y_max)

    # ---------------------------------------------------------
    # Labels and title
    # ---------------------------------------------------------

    plt.xlabel('Yield (kt)')
    plt.ylabel('Height of Burst AGL (m)')

    plt.title(
        f'Calculated HOB vs. Yield\n'
        f'({shock_model.capitalize()} Shock Model, '
        f'Terrain = {terr_alt_m:.1f} m MSL)'
    )

    # ---------------------------------------------------------
    # Measurement annotation
    # ---------------------------------------------------------

    plt.text(
        0.66,
        0.02,
        f'Measured arrival time = {t_meas:.3f} s',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(
            facecolor='white',
            alpha=0.8
        )
    )

    # ---------------------------------------------------------
    # Classification markings
    # ---------------------------------------------------------

    plt.gcf().text(
        .06,
        .95,
        'UNCLASSIFIED',
        fontsize=10,
        color='green'
    )

    plt.gcf().text(
        .78,
        .05,
        'UNCLASSIFIED',
        fontsize=10,
        color='green'
    )

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------

    plt.legend(
        loc='best',
        fontsize=9
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    pdf_pages.savefig(fig)

    st.pyplot(
        fig,
        use_container_width=False
    )

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