# Flash to Shock Analysis Code
# C.J. Miko, 29 December 2025
# This code calculates the range of yields and best yield for an explosive event,
# given distance and time measurements.
# Adapted from the blast wave model from Wei & Hargather (2021, 2023)
# Also includes digitized radius vs time measurements for a surface event
# from Glasstone and Dolan (1977)
# This code is UNCLASSIFIED.

import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

# initial setup
basename = f"/Users/julie/Desktop/Projects/FlashtoShock/"

# constants
ftm = 0.3048    # conversion factor from feet to meters
ktj = 4.184e+12 # conversion factor from kt to joules
P0  = 101325.   # sea-level atmospheric pressure
H   = 8000.     # scale height of atmosphere (approx in meters)
R_s = 287.05    # specific gas constant (J / kg*K)
gam = 1.4       # adiabatic constant for diatomic gas

# define empirical coefficients for Wei-Hargather weak shock solution
u = 0.45   # 0.45 is the literature value
v = 1.    # 1 is the literature value
w = 15.   # 15 is the literature value
p = 1.4  # 1.4 is the literature value
q = 0.3  # 0.3 is the literature value

delta_d   = 1.e6   # m initial value
delta_dm  = 1.e6   # m initial value
delta_dp  = 1.e6   # m initial value

#### Glasstona and Dolan 1 kt scaled surface shock radius vs time ####

# distance (radius) data for 1 kt surface shot from G&D - do not edit
r_list = [0.092542, 0.116779, 0.144321, 0.171863, 0.200507, 0.225846, 0.251185, 0.291947, 0.336014, 0.376777,
          0.458549, 0.52868, 0.722888, 0.906308, 1.078938, 1.251568, 1.424198, 1.596828, 1.769458]

# time data for 1 kt surface shot from G&D - do not edit
t_list = [0.04, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6,
          0.8, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5]

############

#### plotting functions ####

# function to plot back-calculated distance values vs yield
def plot_d_vs_yield(W_guesses, d_calc_array, dp_array, dm_array, d_meas, d_sigmaplus1, d_sigmaminus1,
                    W_best, W_min, W_max, FA, SFC, strong, weak):

    fig = plt.figure(figsize=(9.0, 6.5))

    plt.gcf().text(.14, .90, 'Back-Calculated Distance vs Yield Plot:', fontsize=12, color='black')

    # perform free air/surface logic to modify yield arrays
    if(FA == True):
        W_guesses = W_guesses
        W_best    = W_best
        W_min     = W_min
        W_max     = W_max
        plt.gcf().text(.52, .90, 'Free Air Explosion', fontsize=12, color='black')
    if(SFC == True):
        W_guesses = 0.5 * W_guesses
        W_best    = 0.5 * W_best
        W_min     = 0.5 * W_min
        W_max     = 0.5 * W_max
        plt.gcf().text(.52, .90, 'Surface Explosion', fontsize=12, color='black')

    if(strong == True):
        plt.gcf().text(.72, .90, 'Taylor Solution', fontsize=12, color='black')
    if(weak == True):
        plt.gcf().text(.72, .90, 'W-H Solution', fontsize=12, color='black')

    xmin = 0.8 * W_guesses.min()
    xmax = 1.2 * W_guesses.max()
    ymin = 0.8 * d_meas
    ymax = 1.2 * d_meas
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xscale('log')

    lower_error = dm_array
    upper_error = dp_array
    asymmetric_error = [lower_error, upper_error]

    plt.scatter(W_guesses, d_calc_array, marker='o', s=5, color='blue', label='Back-Calculated Distances')
    plt.errorbar(W_guesses, d_calc_array, yerr=asymmetric_error, fmt='none', ecolor='blue', capsize=5)
    plt.xlabel('Yield (kt)')
    plt.ylabel('Distance (meters)')

    plt.axhline(d_meas, linestyle='-', color='green', label=f'Measured Distance = {d_meas:.1f} m')
    plt.axhline(d_sigmaplus1, linestyle='--', color='green', label=f'Meas Distance+1σ = {d_sigmaplus1:.1f} m')
    plt.axhline(d_sigmaminus1, linestyle='--', color='green', label=f'Meas Distance-1σ = {d_sigmaminus1:.1f} m')

    plt.axvline(W_best, linestyle='--', color='red', label=f'Best Yield = {W_best:.3f} kt')
    plt.axvline(W_min, linestyle='--', color='orange', label=f'Min Yield = {W_min:.3f} kt')
    plt.axvline(W_max, linestyle='--', color='orange', label=f'Max Yield = {W_max:.3f} kt')

    plt.gcf().text(.06, .95, 'UNCLASSIFIED', fontsize=10, color='green')
    plt.gcf().text(.78, .05, 'UNCLASSIFIED', fontsize=10, color='green')

    plt.legend(loc='lower right')
    plt.grid(True)
    pdf_pages.savefig(fig)


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
    
    pdf_pages.savefig(fig)

#### I/O and problem setup functions ####

# writing out initial parameters function
def write_init_params(d_meas, d_meas_unc, d_sigmaplus1, d_sigmaminus1, t_meas, t_meas_unc, W_low, W_high, nguesses, P, T, rho, a, W_guesses):

    # opening initial parameters output file
    init_params = open(f'{basename}init_params.owt', 'a')

    # writing initial parameters
    init_params.write(f"Initial Parameters Output File\n")
    init_params.write(f"\n")
    init_params.write(f"Measured distance (m):                {d_meas:.1f}\n")
    init_params.write(f"Measured Distance Uncertainty (m):    {d_meas_unc:.1f}\n")
    init_params.write(f"Meas Distance + 1-sigma (meters):     {d_sigmaplus1:.1f}\n")
    init_params.write(f"Meas Distance - 1-sigma (meters):     {d_sigmaminus1:.1f}\n")
    init_params.write(f"Measured flash to shock time (sec):   {t_meas:.2f}\n")
    init_params.write(f"Measured time uncertainty (sec):      {t_meas_unc:.2f}\n")
    init_params.write(f"Lowest Possible Yield Guess (kt):     {W_low:.3f}\n")
    init_params.write(f"Highest Possible Yield Guess (kt):    {W_high:.3f}\n")
    init_params.write(f"\n")
    init_params.write(f"Ambient atmospheric pressure (N/m^2): {P:.1f}\n")
    init_params.write(f"Ambient atmospheric temperature (K):  {T:.2f}\n")
    init_params.write(f"Ambient atmospheric density (kg/m^3): {rho:.3f}\n")
    init_params.write(f"Ambient Sound Speed (m/sec):          {a:.2f}\n")
    init_params.write(f"\n")
    init_params.write(f"Number of random yield guesses:       {nguesses}\n")
    init_params.write(f"Random Yield Guesses Array (kt):\n")

    for i in range(0,nguesses,1):
        init_params.write(f"{W_guesses[i]:.3f}\n")

    init_params.close()

    return

def write_final_results(FA, SFC, W_best, W_min, W_max, outputs):

    if(FA == True):
        print(f"# For free air burst:")
        print(f"# Best Yield (kt): {W_best:.3f}")
        print(f"# Min Yield (kt):  {W_min:.3f}")
        print(f"# Max Yield (kt):  {W_max:.3f}")
    if(SFC == True):
        print(f"# For surface burst:")
        print(f"# Best Yield (kt): {0.5 * W_best:.3f}")
        print(f"# Min Yield (kt):  {0.5 * W_min:.3f}")
        print(f"# Max Yield (kt):  {0.5 * W_max:.3f}")
    print(f"###############################################")

    # write out out best, min, and max nuclear yields
    outputs.write(f"###############################################\n")
    outputs.write(f"# Results:\n")
    if(FA == True):
        outputs.write(f"# For free air burst:\n")
        outputs.write(f"# Best Yield (kt): {W_best:.3f}\n")
        outputs.write(f"# Min Yield (kt):  {W_min:.3f}\n")
        outputs.write(f"# Max Yield (kt):  {W_max:.3f}\n")
    if(SFC == True):
        outputs.write(f"# For surface burst:\n")
        outputs.write(f"# Best Yield (kt): {0.5 * W_best:.3f}\n")
        outputs.write(f"# Min Yield (kt):  {0.5 * W_min:.3f}\n")
        outputs.write(f"# Max Yield (kt):  {0.5 * W_max:.3f}\n")
    outputs.write(f"###############################################\n")

    return


# function to set up output arrays for yield assessment
def setup_output_arrays(W_low, W_high, nguesses):

    W_guesses = np.random.uniform(low=W_low, high=W_high, size=nguesses)
    # print(f"Yield guesses in kt: {W_guesses}\n")

    d_calc_list       = []
    dp_list           = []
    dm_list           = []

    return W_guesses, d_calc_list, dp_list, dm_list

############

#### Analytical functions ####

def calc_atmos_properties(T, alt):

    # first, convert temperature to Kelvin
    T = T + 273.15
    print(f"Temperature in Kelvin: {T:.3f}")

    # next, calculate atmospheric pressure at given altitude (meters)
    P = P0 * math.exp(-alt / H)
    print(f"Pressure in Pa:        {P:.3f}")

    # now use ideal gas law to find atmospheric density at given T and alt
    # PV = nRT
    # P  = rho * R_s * T
    # rho = P / (R_s * T)
    rho = P / (R_s * T)
    print(f"density in kg/m^3:     {rho:.3f}")

    # next, calculate sound speed
    a = np.sqrt((gam * P) / rho)
    print(f"Sound speed in m/sec:  {a:.3f}")

    return P, T, rho, a

# function to convert distance measurements and calculate uncertainties
def calc_distance_values(t_meas, t_meas_unc, d_meas, d_meas_unc):

    # convert from ft to meters
    d_meas     = ftm * d_meas
    d_meas_unc = ftm * d_meas_unc
    print(f"Measured distance in meters:            {d_meas:.3f}")
    print(f"Distance uncertainty in meters:         {d_meas_unc:.3f}")
    print(f"Measured flash-to-shock time in sec:    {t_meas:.3f}")
    print(f"Flash-to-shock time uncertainty in sec: {t_meas_unc:.3f}\n")

    # calculate +1 and -1 sigma values
    d_sigmaplus1  = d_meas + d_meas_unc
    d_sigmaminus1 = d_meas - d_meas_unc
    print(f"One StDev above measured distance:      {d_sigmaplus1:.3f} meters")
    print(f"One StDev below measured distance:      {d_sigmaminus1:.3f} meters")

    return d_meas, d_meas_unc, d_sigmaplus1, d_sigmaminus1     


def scaled_radius_time_calc(W_guess, rho, a, t_meas, t_meas_unc, u, v, w, p, q,
                            outputs, strong, weak):

    outputs.write(f"Yield Guess:                        {W_guess:.3f}\n")

    # convert yield in kt to joules
    W_guess_joules = W_guess * ktj

    # define characteristic length and time scales
    lc = (W_guess_joules / (rho * a**2.))**(1. / 3.)  # characteristic length scale
    tc = lc / a                                       # characteristic time scale

    if(strong == True):
        # calculate scaled time and radius values from measured time and distance values
        tstar = t_meas / tc
        rstar = tstar**(2. / 5.)  # Taylor solution
        print(f"Scaled time:                        {tstar:.3e} sec")
        print(f"Scaled radius:                      {rstar:.3e} m")
        outputs.write(f"Scaled time:                        {tstar:.3e} sec\n")
        outputs.write(f"Scaled radius:                      {rstar:.3e} m\n")

        tstar_plus = (t_meas + t_meas_unc) / tc
        rstar_plus = tstar_plus**(2. / 5.)  # Taylor solution
        print(f"Scaled time + 1 sigma:              {tstar_plus:.3e} sec")
        print(f"Scaled radius + 1 sigma:            {rstar_plus:.3e} m")
        outputs.write(f"Scaled time + 1 sigma:              {tstar_plus:.3e} sec\n")
        outputs.write(f"Scaled radius + 1 sigma:            {rstar_plus:.3e} m\n")

        tstar_minus = (t_meas - t_meas_unc) / tc
        rstar_minus = tstar_minus**(2. / 5.)  # Taylor solution
        print(f"Scaled time - 1 sigma:              {tstar_minus:.3e} sec")
        print(f"Scaled radius - 1 sigma:            {rstar_minus:.3e} m")
        outputs.write(f"Scaled time - 1 sigma:              {tstar_minus:.3e} sec\n")
        outputs.write(f"Scaled radius - 1 sigma:            {rstar_minus:.3e} m\n")

    if(weak == True):
        # calculate scaled time and radius values from measured time and distance values
        tstar = t_meas / tc
        rstar = tstar + u * (np.log(v + w * (tstar)**p))**q
        print(f"Scaled time:                        {tstar:.3e} sec")
        print(f"Scaled radius:                      {rstar:.3e} m")
        outputs.write(f"Scaled time:                        {tstar:.3e} sec\n")
        outputs.write(f"Scaled radius:                      {rstar:.3e} m\n")

        tstar_plus = (t_meas + t_meas_unc) / tc
        rstar_plus = tstar_plus + u * (np.log(v + w * (tstar_plus)**p))**q
        print(f"Scaled time + 1 sigma:              {tstar_plus:.3e} sec")
        print(f"Scaled radius + 1 sigma:            {rstar_plus:.3e} m")
        outputs.write(f"Scaled time + 1 sigma:              {tstar_plus:.3e} sec\n")
        outputs.write(f"Scaled radius + 1 sigma:            {rstar_plus:.3e} m\n")

        tstar_minus = (t_meas - t_meas_unc) / tc
        rstar_minus = tstar_minus + u * (np.log(v + w * (tstar_minus)**p))**q
        print(f"Scaled time - 1 sigma:              {tstar_minus:.3e} sec")
        print(f"Scaled radius - 1 sigma:            {rstar_minus:.3e} m")
        outputs.write(f"Scaled time - 1 sigma:              {tstar_minus:.3e} sec\n")
        outputs.write(f"Scaled radius - 1 sigma:            {rstar_minus:.3e} m\n")

    return rstar, tstar, rstar_plus, tstar_plus, rstar_minus, tstar_minus, lc, tc


# function to back-calculate distance from scaled distance value
def d_calc(rstar, rstar_plus, rstar_minus, lc, outputs):

    # calculate true radius (distance) from previously-calculated scaled distance
    d       = rstar * lc
    d_plus  = rstar_plus * lc
    d_minus = rstar_minus * lc

    # calculate positive and negative uncertainties
    dp = d_plus - d
    dm = d - d_minus

    print(f"Back-calculated distance:           {d:.3f} m")
    print(f"Back-calculated distance + 1 sigma: {d_plus:.3f} m")
    print(f"Back-calculated distance - 1 sigma: {d_minus:.3f} m\n")
    outputs.write(f"Back-calculated distance:           {d:.3f} m\n")
    outputs.write(f"Back-calculated distance + 1 sigma: {d_plus:.3f} m\n")
    outputs.write(f"Back-calculated distance - 1 sigma: {d_minus:.3f} m\n\n")

    return d, d_plus, d_minus, dp, dm

############


############################################################################
#                             Main Program                                 #
############################################################################
d_meas       = 2380.  # measured distance in feet
d_meas_unc   = 20.    # measured distance uncertainty in feet
t_meas       = 1.75   # measured flash to shock time in sec
t_meas_unc   = 1/30   # measured timing uncertainty in sec
W_low        = 0.03   # lowest guess in kt yield
W_high       = 0.5    # highest guess in kt yield
nguesses     = 1000  # size of random yield array (number of guesses)
T            = 25.    # temperature in Celsius
alt          = 0.     # altitude above MSL in meters

# free air vs surface options
FA  = False
SFC = True

# Taylor strong shock-only vs Wei & Hargather weak shock
strong = False
weak   = True

# setting output file path and opening outputs file and PDF
outputs = open(f'{basename}outputs.owt', 'a')
owtname = basename + f'/WH_Outputs_Plot.pdf'
pdf_pages = PdfPages(owtname)

#### beginning calculations ####
print(f"Beginning Calculations to determine yield.\n")
outputs.write(f"Outputs Data File\n\n")
outputs.write(f"Calculations to determine yield\n\n")

# convert measured distance in feet to meters and print out
d_meas, d_meas_unc, d_sigmaplus1, d_sigmaminus1 = calc_distance_values(t_meas, t_meas_unc, d_meas, d_meas_unc)

# set up random yield guesses array and back-calculated distance list
W_guesses, d_calc_list, dp_list, dm_list = setup_output_arrays(W_low, W_high, nguesses)

# calculate atmospheric density and sound speed
P, T, rho, a = calc_atmos_properties(T, alt)

# opening outputs file
write_init_params(d_meas, d_meas_unc, d_sigmaplus1, d_sigmaminus1, t_meas, t_meas_unc, W_low, W_high, nguesses, P, T, rho, a, W_guesses)

for i in range(0,nguesses,1):

    print(f"Yield guess:                        {W_guesses[i]:.3f} kt:")

    # find scaled radius and time from Wei-Hargather blast wave solution
    rstar, tstar, rstar_plus, tstar_plus, rstar_minus, tstar_minus, lc, tc = scaled_radius_time_calc(W_guesses[i], rho, a, t_meas, t_meas_unc,
                                                                                                         u, v, w, p, q, outputs, strong, weak)

    # back-calculate true radius (distance)
    d, d_plus, d_minus, dp, dm = d_calc(rstar, rstar_plus, rstar_minus, lc, outputs)

    # append lists
    d_calc_list.append(d)
    dp_list.append(dp)
    dm_list.append(dm)

    # assign best values for back-calculated distance and yield values
    if(abs(d - d_meas) < delta_d):
        delta_d = abs(d - d_meas)
        W_best = W_guesses[i]
    if(abs(d_plus - d_sigmaminus1) < delta_dm):
        delta_dm = abs(d_plus - d_sigmaminus1)
        W_min = W_guesses[i]
    if(abs(d_sigmaplus1 - d_minus) < delta_dp):
        delta_dp = abs(d_sigmaplus1 - d_minus)
        W_max = W_guesses[i]

# generate back-calculated distance arrays and uncertainties
d_calc_array = np.array(d_calc_list)
dp_array     = np.array(dp_list)
dm_array     = np.array(dm_list)

# apply free air/surface options and output:
write_final_results(FA, SFC, W_best, W_min, W_max, outputs)
print("STOP all done flash to shock yield calculations complete.")

# plot nuclear detonation distance vs yield plot
plot_d_vs_yield(W_guesses, d_calc_array, dp_array, dm_array, d_meas, d_sigmaplus1, d_sigmaminus1,
                W_best, W_min, W_max, FA, SFC, strong, weak)

# plot fit of strong, weak, and acoustic radius vs time
plot_fit(W_best, rho, a, d_meas, t_meas, u, v, w, p, q)

outputs.close()
pdf_pages.close()
print('Plotting complete.')