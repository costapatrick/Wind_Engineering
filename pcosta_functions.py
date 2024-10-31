# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:28:42 2022

@author: saopabat
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import gamma
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.ticker as ticker

def linear_regression(V_ord, y, model):
    """Fit a linear regression model and return coefficients and metrics."""
    model.fit(y, V_ord)
    a = float(model.coef_)
    b = float(model.intercept_)
    
    # Calculate metrics
    predictions = model.predict(y)
    r_squared = r2_score(V_ord, predictions)
    mae = mean_absolute_error(V_ord, predictions)
    mse = mean_squared_error(V_ord, predictions)
    rmse = np.sqrt(mse)

    return a, b, r_squared, mae, mse, rmse, predictions


def plot_regression(y, V_ord, model_predictions, a, b, r_squared, title, filename, figure_number):
    """Plot linear regression results."""
    plt.figure(figure_number, dpi=400)
    plt.scatter(y, V_ord, color='blue')
    plt.plot(y, model_predictions, color='red')
    plt.text(y[7], V_ord[-2], f'{a:.3f} x + {b:.3f}', fontsize=12, color='black', fontweight='bold')
    
    bbox_props = dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black")
    plt.text(y[-2], V_ord[2], f'R² = {r_squared:.2f}', fontsize=12, color='black', fontweight='bold', bbox=bbox_props)
    
    plt.title(title)
    plt.xlabel('Y')
    plt.ylabel('Maximum annual velocity (m/s) - 10 min')
    plt.grid()
    plt.savefig(filename, dpi=300., bbox_inches='tight')
    # plt.close()


def calculate_probability_and_y(Vmax):
    """Calculate probability and y values for regression."""
    V_ord = np.sort(Vmax)
    M = np.argsort(V_ord) + 1
    Px = M / (M[-1] + 1)
    y = (-np.log(-np.log(Px))).reshape(-1, 1)
    return Px, y, V_ord


def raw_regression(Vmax, dist='Gumbel', units='m/s'):
    """Perform linear regression on raw data."""
    if dist == 'Gumbel' and units == 'm/s':
        Px, y, V_ord = calculate_probability_and_y(Vmax)
        model = LinearRegression()
        
        a, b, r_squared, mae, mse, rmse, predictions = linear_regression(V_ord, y, model)
        
        plot_regression(y, V_ord, predictions, a, b, r_squared, 'Linear Regression - Raw Data', 'Plots/Raw_Regression.png', 1)
        
        print_metrics(r_squared, mae, mse, rmse)
    
    return Px, y, V_ord, a, b


def process_data(Vmax, val):
    """Remove outliers from the data."""
    mean_y = np.mean(Vmax)
    std_y = np.std(Vmax)
    threshold = val * std_y
    outlier_indices = np.where(np.abs(Vmax - mean_y) > threshold)[0]
    return np.delete(Vmax, outlier_indices)


def processed_regression(Vmax, val=2, dist='Gumbel', units='m/s'):
    """Perform linear regression on processed data."""
    if dist == 'Gumbel' and units == 'm/s':
        y_cleaned = process_data(Vmax, val)
        Px, y, V_ord = calculate_probability_and_y(y_cleaned)
        model = LinearRegression()
        
        a, b, r_squared, mae, mse, rmse, predictions = linear_regression(V_ord, y, model)

        plot_regression(y, V_ord, predictions, a, b, r_squared, 'Linear Regression - Processed Data', 'Plots/Processed_Regression.png', 2)

        print_metrics(r_squared, mae, mse, rmse)
    
    return Px, y, V_ord, a, b


def robust_regression(Vmax, dist='Gumbel', units='m/s'):
    """Perform robust linear regression."""
    if dist == 'Gumbel' and units == 'm/s':
        Px, y, V_ord = calculate_probability_and_y(Vmax)
        model = HuberRegressor()

        a, b, r_squared, mae, mse, rmse, predictions = linear_regression(V_ord, y, model)

        plot_regression(y, V_ord, predictions, a, b, r_squared, 'Linear Regression - Robust Data', 'Plots/Robust_Regression.png', 3)

        print_metrics(r_squared, mae, mse, rmse)
    
    return Px, y, V_ord, a, b


def print_metrics(r_squared, mae, mse, rmse):
    """Print evaluation metrics."""
    print("R² Score:", r_squared)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("RMSE:", rmse)
    print()
    print()


def calc_Vo(a, b, T, t_Vmax='600'):
    """
    Valid for Vo in category II
    t_Vmax: is the duration of the velocity measurement in seconds [s]
    """
    T = np.array(T)
    PR = 1 - 1 / T 

    # Use np.where to avoid taking log of zero or negative values
    valid_indices = PR > 0  # We only want to proceed with positive PR values

    if np.any(valid_indices):
        yR = np.zeros_like(T, dtype=float)  # Initialize yR array with zeros
        yR[valid_indices] = np.log(T[valid_indices]) - np.log(-T[valid_indices] * np.log(PR[valid_indices]))

        # Handle any invalid (i.e., negative or zero) cases
        yR[~valid_indices] = 0  # Set invalid yR values to 0

        Vo_10min = a * yR + b

        # Adjust Vo_3s based on t_Vmax
        t_Vmax_dict = {
            '5': 0.98, '10': 0.95, '15': 0.93,
            '20': 0.90, '30': 0.87, '45': 0.84,
            '60': 0.82, '120': 0.77, '300': 0.72,
            '600': 0.69, '3600': 0.65
        }
        
        Vo_3s = Vo_10min / t_Vmax_dict.get(t_Vmax, 1)  # Default to 1 if t_Vmax not found
    else:
        Vo_10min = np.zeros_like(T)  # Initialize to zero if no valid PR values
        Vo_3s = np.zeros_like(T)

    plt.figure(4, dpi=400)
    plt.semilogx(T, Vo_3s)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.xlabel('Return Period - T (years)', fontweight='bold')
    plt.ylabel('Vo (m/s)', fontweight='bold')
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.xlim(T[0], T[-1])

    plt.savefig('Plots/Vo_velocity.png', dpi=300., bbox_inches='tight')

    for i in range(len(T)):
        print('The basic wind speed (Vo) for a return period (T) of %d year is: %.2f m/s' % (T[i], Vo_3s[i]))

    return Vo_10min, Vo_3s
