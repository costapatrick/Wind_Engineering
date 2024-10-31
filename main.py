# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:45:22 2022
@author: pcosta
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from pcosta_functions import raw_regression, processed_regression, robust_regression, calc_Vo

def load_data(file_path):
    """Load maximum annual wind speed data from a file."""
    try:
        return np.loadtxt(file_path)
    except IOError as e:
        raise IOError(f"File '{file_path}' not found or cannot be read.") from e

def main():
    # Load maximum annual wind speed data in m/s
    Vmax = load_data('poa.txt')

    # Define return periods (in years)
    return_periods = [1, 10, 35, 50, 100, 1000]

    # Basic statistical calculations
    mean_wind = Vmax.mean()
    std_wind = Vmax.std()
    cv_wind = std_wind / mean_wind

    # Calculate parameters using various regression models
    Px, y, V_ordered, a, b = raw_regression(Vmax, dist='Gumbel', units='m/s')
    Pxp, yp, V_ordered_p, a_p, b_p = processed_regression(Vmax, val=3, dist='Gumbel', units='m/s')
    Pxr, yr, V_ordered_r, a_r, b_r = robust_regression(Vmax, dist='Gumbel', units='m/s')

    # Calculate wind speeds for 10-minute and 3-second intervals
    Vo_10min, Vo_3s = calc_Vo(a, b, return_periods, t_Vmax='600')

if __name__ == "__main__":
    main()



#%%

# ###############################################################################
# #                           S3 FACTOR OF NBR                                  #
# ###############################################################################

# # Compute shape parameter beta for Frechet distribution
# if cv_wind > 0.233:
#     cv_function = lambda beta_val: np.sqrt(gamma(1. - 2. / beta_val) - (gamma(1. - 1. / beta_val))**2) / gamma(1. - 1. / beta_val)
#     beta = 6.37
#     error_tolerance = 0.001
#     error = 1.0

#     while abs(error) > error_tolerance:
#         delta = cv_function(beta)
#         error = cv_wind - delta
#         beta += 0.01 if error < 0 else -0.01
# else:
#     beta = 6.37

# # Calculate scale parameter alpha for Frechet distribution
# alpha = mean_wind / gamma(1. - 1. / beta)

# # Determine Vk for m = 50 years with probability Pm = 63%
# m_years = 50
# Pm_63 = 1 - np.exp(-1)
# Vk = alpha * (-np.log(1 - Pm_63) / m_years) ** (-1 / beta)

# # Calculate S3 factors using Frechet and Gumbel distributions
# S3_frechet = 0.54 * (-np.log(1 - Pm_63) / np.array(return_periods)) ** -0.157
# S3_gumbel = (7 - np.log(-np.log(1 - Pm_63) / np.array(return_periods))) * (1 / 10.9)

# ###############################################################################
# #                                PLOTTING                                     #
# ###############################################################################

# # Configure plot with S3 factors for different return periods
# plt.figure(figsize=(10, 6))
# plt.semilogx(return_periods, S3_frechet, color='blue', linestyle='--', label='Fisher-Tippett type II (Fréchet)')
# plt.semilogx(return_periods, S3_gumbel, color='red', label='Fisher-Tippett type I (Gumbel)')

# # Plot annotations for specific return periods (50 and 35 years)
# highlight_periods = [50, 35]
# annotation_offset = 0.07

# for period in highlight_periods:
#     idx = return_periods.index(period)
#     # Frechet values
#     plt.text(period, S3_frechet[idx] - annotation_offset, f'{S3_frechet[idx]:.2f}', fontsize=10, color='black', fontweight='bold')
#     plt.plot([0, period], [S3_frechet[idx]] * 2, 'k:')
#     plt.plot([period] * 2, [0.5, S3_frechet[idx]], 'k:')
#     plt.plot(period, S3_frechet[idx], 'ko')

#     # Gumbel values
#     plt.text(period - 8, S3_gumbel[idx] + annotation_offset, f'{S3_gumbel[idx]:.2f}', fontsize=10, color='red', fontweight='bold')
#     # plt.plot([0, period], [S3_gumbel[idx]] * 2, 'ko:')
#     plt.plot(period, S3_gumbel[idx], marker='o', color='red')
#     plt.plot([period] * 2, [0.5, S3_gumbel[idx]], 'k:')
#     plt.plot(period, S3_gumbel[idx], 'ko', color='red')

# # Customize plot appearance
# plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
# plt.xlabel('Return Period - T (years)', fontweight='bold')
# plt.ylabel('S3', fontweight='bold')
# plt.legend()
# plt.grid(which='both', linestyle='--', linewidth=0.5)
# plt.xlim(return_periods[0], return_periods[-1])
# plt.ylim(0.5, 1.7)

# # Save and display the plot
# plt.savefig('Plots/S3_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()
