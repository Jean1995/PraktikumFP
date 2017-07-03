##################################################### Import system libraries ######################################################
import matplotlib as mpl
mpl.rcdefaults()
mpl.rcParams.update(mpl.rc_params_from_file('meine-matplotlibrc'))
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)
################################################ Finish importing system libraries #################################################

################################################ Adding subfolder to system's path #################################################
import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

 # use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"python_custom_scripts")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
############################################# Finish adding subfolder to system's path #############################################

##################################################### Import custom libraries ######################################################
from curve_fit import ucurve_fit
from table import (
    make_table,
    make_full_table,
    make_composed_table,
    make_SI,
    write,
)
from regression import (
    reg_linear,
    reg_quadratic,
    reg_cubic
)
from error_calculation import(
    MeanError
)
################################################ Finish importing custom libraries #################################################

from scipy.optimize import curve_fit
from numpy import random

from scipy.interpolate import spline

plt.style.use('seaborn-darkgrid')
plt.set_cmap('Set2')




################################ FREQUENTLY USED CODE ################################
#
########## IMPORT ##########
# t, U, U_err = np.genfromtxt('data.txt', unpack=True)
# t *= 1e-3


########## ERRORS ##########
# R_unc = ufloat(R[0],R[2])
# U = 1e3 * unp.uarray(U, U_err)
# Rx_mean = np.mean(Rx)                 # Mittelwert und syst. Fehler
# Rx_mean_err = MeanError(noms(Rx))     # Fehler des Mittelwertes
#
## Relative Fehler zum späteren Vergleich in der Diskussion
# RelFehler_G = (G_mess - G_lit) / G_lit
# RelFehler_B = (B_mess - B_lit) / B_lit
# write('build/RelFehler_G.tex', make_SI(RelFehler_G*100, r'\percent', figures=1))
# write('build/RelFehler_B.tex', make_SI(RelFehler_B*100, r'\percent', figures=1))


########## CURVE FIT ##########
# def f(t, a, b, c, d):
#     return a * np.sin(b * t + c) + d
#
# params = ucurve_fit(f, t, U, p0=[1, 1e3, 0, 0])   # p0 bezeichnet die Startwerte der zu fittenden Parameter
# params = ucurve_fit(reg_linear, x, y)             # linearer Fit
# params = ucurve_fit(reg_quadratic, x, y)          # quadratischer Fit
# params = ucurve_fit(reg_cubic, x, y)              # kubischer Fit
# a, b = params
# write('build/parameter_a.tex', make_SI(a * 1e-3, r'\kilo\volt', figures=1))       # type in Anz. signifikanter Stellen
# write('build/parameter_b.tex', make_SI(b * 1e-3, r'\kilo\hertz', figures=2))      # type in Anz. signifikanter Stellen


########## PLOTTING ##########
# plt.clf                   # clear actual plot before generating a new one
#
## automatically choosing limits with existing array T1
# t_plot = np.linspace(np.amin(T1), np.amax(T1), 100)
# plt.xlim(t_plot[0]-1/np.size(T1)*(t_plot[-1]-t_plot[0]), t_plot[-1]+1/np.size(T1)*(t_plot[-1]-t_plot[0]))
#
## hard coded limitsmarkersize=20,
# t_plot = np.linspace(-0.5, 2 * np.pi + 0.5, 1000) * 1e-3
#
## standard plotting
# plt.plot(t_plot * 1e3, f(t_plot, *noms(params)) * 1e-3, 'b-', label='Fit')
# plt.plot(t * 1e3, U * 1e3, 'rx', label='Messdaten')
## plt.errorbar(B * 1e3, noms(y) * 1e5, fmt='rx', yerr=stds(y) * 1e5, label='Messdaten')        # mit Fehlerbalken
## plt.xscale('log')                                                                            # logarithmische x-Achse
# plt.xlim(t_plot[0] * 1e3, t_plot[-1] * 1e3)
# plt.xlabel(r'$t \:/\: \si{\milli\second}$')
# plt.ylabel(r'$U \:/\: \si{\kilo\volt}$')
# plt.legend(loc='best')
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/aufgabenteil_a_plot.pdf')


########## WRITING TABLES ##########
### IF THERE IS ONLY ONE COLUMN IN A TABLE (workaround):
## a=np.array([Wert_d[0]])
## b=np.array([Rx_mean])
## c=np.array([Rx_mean_err])
## d=np.array([Lx_mean*1e3])
## e=np.array([Lx_mean_err*1e3])
#
# write('build/Tabelle_b.tex', make_table([a,b,c,d,e],[0, 1, 0, 1, 1]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
# write('build/Tabelle_b_texformat.tex', make_full_table(
#     'Messdaten Kapazitätsmessbrücke.',
#     'table:A2',
#     'build/Tabelle_b.tex',
#     [1,2,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
#                               # die Multicolumns sein sollen
#     ['Wert',
#     r'$C_2 \:/\: \si{\nano\farad}$',
#     r'$R_2 \:/\: \si{\ohm}$',
#     r'$R_3 / R_4$', '$R_x \:/\: \si{\ohm}$',
#     r'$C_x \:/\: \si{\nano\farad}$']))
#
## Aufsplitten von Tabellen, falls sie zu lang sind
# t1, t2 = np.array_split(t * 1e3, 2)
# U1, U2 = np.array_split(U * 1e-3, 2)
# write('build/loesung-table.tex', make_table([t1, U1, t2, U2], [3, None, 3, None]))  # type in Nachkommastellen
#
## Verschmelzen von Tabellen (nur Rohdaten, Anzahl der Zeilen muss gleich sein)
# write('build/Tabelle_b_composed.tex', make_composed_table(['build/Tabelle_b_teil1.tex','build/Tabelle_b_teil2.tex']))


########## ARRAY FUNCTIONS ##########
# np.arange(2,10)                   # Erzeugt aufwärts zählendes Array von 2 bis 10
# np.zeros(15)                      # Erzeugt Array mit 15 Nullen
# np.ones(15)                       # Erzeugt Array mit 15 Einsen
#
# np.amin(array)                    # Liefert den kleinsten Wert innerhalb eines Arrays
# np.argmin(array)                  # Gibt mir den Index des Minimums eines Arrays zurück
# np.amax(array)                    # Liefert den größten Wert innerhalb eines Arrays
# np.argmax(array)                  # Gibt mir den Index des Maximums eines Arrays zurück
#
# a1,a2 = np.array_split(array, 2)  # Array in zwei Hälften teilen
# np.size(array)                    # Anzahl der Elemente eines Arrays ermitteln


########## ARRAY INDEXING ##########
# y[n - 1::n]                       # liefert aus einem Array jeden n-ten Wert als Array


########## DIFFERENT STUFF ##########
# R = const.physical_constants["molar gas constant"]      # Array of value, unit, error


### Bisschen Messdaten und so

# I = 0 (kein Magnetfeld), T=195°C

#erster Punkt (0, 0.035) rausgenommen
l0 = np.array([4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.75, 5.8, 5.85, 5.9, 5.95, 6, 6.05, 6.06, 6.07, 6.1, 6.15, 6.2, 6.22, 6.24, 6.26, 6.28, 6.3, 6.32, 6.34, 6.36, 6.38, 6.4, 6.42, 6.44, 6.46, 6.48, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.5, 8.0, 9.0 ])
I0 = np.array([  0.06, 0.061, 0.061, 0.063, 0.065, 0.07, 0.072, 0.073, 0.076, 0.077, 0.079, 0.082, 0.091, 0.096, 0.102, 0.112, 0.13, 0.163, 0.193, 0.24, 0.293, 0.348, 0.424, 0.496, 0.571, 0.593, 0.607, 0.652, 0.733, 0.811, 0.843, 0.873, 0.901, 0.924, 0.949, 0.969, 0.983, 0.989, 0.998, 1.003, 1.002, 0.996, 0.986, 0.972, 0.951, 0.806, 0.625, 0.467, 0.329, 0.241, 0.187, 0.106, 0.087, 0.077 ])

# I=0.3 A , T=197°C

l1 = np.array([ 9.0, 8.0, 7.5, 7.0, 6.9, 6.8, 6.7, 6.68, 6.66, 6.64, 6.62, 6.6, 6.58, 6.56, 6.54, 6.52, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.88, 5.86, 5.84, 5.82, 5.8, 5.78, 5.76, 5.74, 5.72, 5.7, 5.68, 5.66, 5.64, 5.62, 5.6, 5.5, 5.4, 5.3, 5.2, 5.1, 5.0, 4.0, 3.0, 2.0 ])
I1 = np.array([ 0.084, 0.111, 0.154, 0.276, 0.333, 0.379, 0.423, 0.432, 0.434, 0.437, 0.442, 0.444, 0.448, 0.449, 0.448, 0.445, 0.443, 0.423, 0.401, 0.383, 0.381, 0.392, 0.423, 0.433, 0.438, 0.445, 0.453, 0.459, 0.468, 0.471, 0.480, 0.483, 0.489, 0.490, 0.489, 0.482, 0.479, 0.473, 0.423, 0.361, 0.304, 0.254, 0.219, 0.191, 0.087, 0.051, 0.040 ])

# I=0.4 A, T= 198°C

l2 = np.array([ 2.0, 3.0, 4.0, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.62, 5.64, 5.66, 5.68, 5.7, 5.72, 5.74, 5.76, 5.78, 5.8, 5.82, 5.84, 5.86, 5.88, 5.9, 5.92, 5.94, 5.96, 5.98, 6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.62, 6.64, 6.66, 6.68, 6.7, 6.72, 6.74, 6.76, 6.78, 6.8, 6.82, 6.84, 6.86, 6.88, 6.9, 6.92, 6.94, 6.96, 6.98, 7.0, 7.02, 7.04, 7.06, 7.08, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 9.0, 10.0 ])
I2 = np.array([ 0.041, 0.048, 0.081, 0.161, 0.172, 0.193, 0.216, 0.247, 0.296, 0.348, 0.361, 0.371, 0.382, 0.389, 0.398, 0.404, 0.413, 0.421, 0.424, 0.429, 0.431, 0.432, 0.434, 0.433, 0.429, 0.425, 0.424, 0.419, 0.412, 0.407, 0.367, 0.325, 0.284, 0.268, 0.280, 0.319, 0.330, 0.339, 0.347, 0.356, 0.368, 0.370, 0.375, 0.383, 0.388, 0.392, 0.396, 0.400, 0.408, 0.407, 0.408, 0.408, 0.412, 0.411, 0.411, 0.411, 0.409, 0.407, 0.402, 0.398, 0.393, 0.359, 0.324, 0.289, 0.251, 0.220, 0.197, 0.178, 0.164, 0.153, 0.089, 0.057 ])

# I=0.5 A, T=198°C

l3 = np.array([ 10.0, 9.0, 8.0, 7.9, 7.8, 7.7, 7.6, 7.5, 7.4, 7.3, 7.2, 7.1, 7.08, 7.06, 7.04, 7.02, 7.0, 6.98, 6.96, 6.94, 6.92, 6.9, 6.88, 6.86, 6.84, 6.82, 6.8, 6.78, 6.76, 6.74, 6.72, 6.7, 6.68, 6.66, 6.64, 6.62, 6.6, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7, 5.68, 5.66, 5.64, 5.62, 5.6, 5.58, 5.56, 5.54, 5.52, 5.5, 5.48, 5.46, 5.44, 5.42, 5.4, 5.3, 5.2, 5.1, 5.0, 4.0, 3.0, 2.0 ])
I3 = np.array([ 0.069, 0.086, 0.143, 0.161, 0.174, 0.190, 0.202, 0.214, 0.236, 0.258, 0.282, 0.311, 0.314, 0.324, 0.327, 0.329, 0.340, 0.346, 0.347, 0.353, 0.357, 0.356, 0.357, 0.363, 0.366, 0.365, 0.366, 0.363, 0.361, 0.359, 0.357, 0.354, 0.352, 0.348, 0.344, 0.340, 0.335, 0.306, 0.272, 0.230, 0.198, 0.181, 0.190, 0.227, 0.282, 0.336, 0.356, 0.363, 0.371, 0.377, 0.381, 0.384, 0.387, 0.390, 0.393, 0.393, 0.392, 0.390, 0.388, 0.384, 0.379, 0.351, 0.321, 0.289, 0.264, 0.115, 0.067, 0.044 ])

# I=0.6A, T = 198°C

l4 = np.array([ 2.0, 3.0, 4.0, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.52, 5.54, 5.56, 5.58, 5.6, 5.62, 5.64, 5.66, 5.68, 5.7, 5.72, 5.74, 5.76, 5.78, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.02, 7.04, 7.06, 7.08, 7.1, 7.12, 7.14, 7.16, 7.18, 7.2, 7.22, 7.24, 7.26, 7.28, 7.3, 7.32, 7.34, 7.36, 7.38, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 9.0, 10.0 ])
I4 = np.array([ 0.051, 0.064, 0.121, 0.211, 0.224, 0.243, 0.271, 0.299, 0.332, 0.335, 0.341, 0.346, 0.353, 0.356, 0.359, 0.366, 0.367, 0.367, 0.363, 0.363, 0.362, 0.358, 0.350, 0.346, 0.316, 0.279, 0.239, 0.199, 0.165, 0.149, 0.162, 0.201, 0.244, 0.282, 0.312, 0.330, 0.332, 0.339, 0.340, 0.336, 0.335, 0.335, 0.340, 0.341, 0.341, 0.339, 0.338, 0.334, 0.333, 0.332, 0.331, 0.329, 0.325, 0.321, 0.317, 0.312, 0.291, 0.267, 0.244, 0.226, 0.213, 0.197, 0.112, 0.075 ])

# I=0.7A, T=198°C

l5 = np.array([ 10.0, 9.0, 8.0, 7.0, 6.98, 6.96, 6.94, 6.92, 6.9, 6.88, 6.86, 6.84, 6.82, 6.8, 6.78, 6.76, 6.74, 6.72, 6.7, 6.6, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7, 5.6, 5.58, 5.56, 5.54, 5.52, 5.5, 5.48, 5.46, 5.44, 5.42, 5.4, 5.38, 5.36, 5.34, 5.32, 5.3, 5.28, 5.26, 5.24, 5.22, 5.2, 5.18, 5.16, 5.14, 5.12, 5.1, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.0, 3.0, 2.0 ])
I5 = np.array([ 0.078, 0.097, 0.164, 0.307, 0.309, 0.310, 0.311, 0.313, 0.311, 0.310, 0.309, 0.309, 0.305, 0.304, 0.306, 0.300, 0.294, 0.291, 0.284, 0.255, 0.224, 0.192, 0.161, 0.134, 0.122, 0.132, 0.156, 0.199, 0.242, 0.287, 0.296, 0.302, 0.313, 0.317, 0.324, 0.327, 0.330, 0.329, 0.333, 0.334, 0.332, 0.335, 0.334, 0.336, 0.331, 0.329, 0.325, 0.322, 0.321, 0.323, 0.319, 0.315, 0.311, 0.309, 0.306, 0.279, 0.260, 0.238, 0.217, 0.198, 0.180, 0.129, 0.071, 0.047 ])

# I=0.8A, T=198°C

l6 = np.array([ 2.0, 3.0, 4.0, 5.0, 5.1, 5.2, 5.22, 5.24, 5.26, 5.28, 5.3, 5.32, 5.34, 5.36, 5.38, 5.4, 5.42, 5.44, 5.46, 5.48, 5.5, 5.52, 5.54, 5.56, 5.58, 5.6, 5.62, 5.64, 5.66, 5.68, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.82, 6.84, 6.86, 6.88, 6.9, 6.92, 6.94, 6.96, 6.98, 7.0, 7.02, 7.04, 7.06, 7.08, 7.1, 7.12, 7.14, 7.16, 7.18, 7.2, 7.22, 7.24, 7.26, 7.28, 7.3, 7.32, 7.34, 7.36, 7.38, 7.4, 7.42, 7.44, 7.46, 7.48, 7.5, 7.52, 7.54, 7.56, 7.58, 7.6, 7.7, 7.8, 7.9, 8.0, 9.0, 10.0 ])
I6 = np.array([ 0.049, 0.066, 0.113, 0.226, 0.244, 0.263, 0.265, 0.267, 0.271, 0.274, 0.278, 0.284, 0.287, 0.289, 0.293, 0.298, 0.299, 0.300, 0.304, 0.309, 0.310, 0.312, 0.314, 0.312, 0.312, 0.311, 0.310, 0.308, 0.305, 0.303, 0.299, 0.268, 0.235, 0.203, 0.176, 0.146, 0.123, 0.116, 0.125, 0.154, 0.185, 0.217, 0.221, 0.228, 0.235, 0.241, 0.243, 0.250, 0.248, 0.254, 0.258, 0.261, 0.266, 0.269, 0.273, 0.276, 0.279, 0.281, 0.282, 0.287, 0.291, 0.291, 0.292, 0.293, 0.291, 0.291, 0.291, 0.293, 0.292, 0.291, 0.295, 0.291, 0.290, 0.289, 0.287, 0.285, 0.281, 0.279, 0.276, 0.273, 0.271, 0.269, 0.258, 0.243, 0.229, 0.216, 0.214, 0.088 ])

#I=0.9A, T=198°C
l7 = np.array([ 10.0, 9.0, 8.0, 7.9, 7.8, 7.7, 7.6, 7.5, 7.4, 7.38, 7.36, 7.34, 7.32, 7.3, 7.28, 7.26, 7.24, 7.22, 7.2, 7.18, 7.16, 7.14, 7.12, 7.1, 7.08, 7.06, 7.04, 7.02, 7.0, 6.98, 6.96, 6.94, 6.92, 6.9, 6.8, 6.7, 6.6, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7, 5.6, 5.58, 5.56, 5.54, 5.52, 5.5, 5.48, 5.46, 5.44, 5.42, 5.4, 5.38, 5.36, 5.34, 5.32, 5.3, 5.28, 5.26, 5.24, 5.22, 5.2, 5.18, 5.16, 5.14, 5.12, 5.1, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.0, 3.0, 2.0 ])
I7 = np.array([ 0.089, 0.109, 0.183, 0.198, 0.208, 0.221, 0.232, 0.246, 0.259, 0.261, 0.262, 0.263, 0.269, 0.270, 0.272, 0.274, 0.275, 0.276, 0.278, 0.280, 0.281, 0.282, 0.282, 0.284, 0.284, 0.284, 0.284, 0.284, 0.281, 0.282, 0.279, 0.278, 0.275, 0.273, 0.259, 0.235, 0.211, 0.182, 0.156, 0.131, 0.116, 0.103, 0.107, 0.123, 0.149, 0.184, 0.224, 0.232, 0.241, 0.248, 0.255, 0.260, 0.267, 0.273, 0.278, 0.280, 0.284, 0.288, 0.291, 0.293, 0.296, 0.298, 0.299, 0.299, 0.303, 0.300, 0.299, 0.298, 0.296, 0.295, 0.294, 0.292, 0.281, 0.265, 0.248, 0.233, 0.221, 0.205, 0.149, 0.087, 0.061 ])

#I=1.0A, T=198°C
l8 = np.array([ 2.0, 3.0, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.12, 5.14, 5.16, 5.18, 5.2, 5.22, 5.24, 5.26, 5.28, 5.3, 5.32, 5.34, 5.36, 5.38, 5.4, 5.42, 5.44, 5.46, 5.48, 5.5, 5.52, 5.54, 5.56, 5.58, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.22, 7.24, 7.26, 7.28, 7.3, 7.32, 7.34, 7.36, 7.38, 7.4, 7.42, 7.44, 7.46, 7.48, 7.5, 7.52, 7.54, 7.56, 7.58, 7.6, 7.7, 7.8, 7.9, 8.0, 9.0, 10.0 ])
I8 = np.array([ 0.061, 0.076, 0.127, 0.134, 0.141, 0.149, 0.160, 0.172, 0.185, 0.199, 0.213, 0.226, 0.237, 0.251, 0.253, 0.253, 0.254, 0.256, 0.259, 0.261, 0.263, 0.265, 0.268, 0.271, 0.272, 0.275, 0.279, 0.280, 0.281, 0.282, 0.285, 0.284, 0.284, 0.284, 0.283, 0.281, 0.277, 0.276, 0.272, 0.250, 0.219, 0.187, 0.159, 0.137, 0.117, 0.101, 0.096, 0.102, 0.121, 0.147, 0.173, 0.203, 0.223, 0.241, 0.258, 0.259, 0.261, 0.265, 0.267, 0.272, 0.273, 0.274, 0.274, 0.274, 0.276, 0.276, 0.275, 0.274, 0.273, 0.274, 0.270, 0.268, 0.265, 0.261, 0.261, 0.252, 0.240, 0.227, 0.216, 0.130, 0.091 ])


plt.plot(l0*1.8, I0, '.', markersize=3,label=r'$I = \SI{0}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\volt}$')
plt.legend(loc='best')
plt.savefig('build/plot0.pdf')
plt.clf()

plt.plot(l1*1.8, I1, '.',markersize=3, label=r'$I = \SI{0.3}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\volt}$')
plt.legend(loc='best')
plt.savefig('build/plot1.pdf')
plt.clf()

plt.plot(l2*1.8, I2, '.',markersize=3, label=r'$I = \SI{0.4}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\volt}$')
plt.legend(loc='best')
plt.savefig('build/plot2.pdf')
plt.clf()

plt.plot(l3*1.8, I3, '.',markersize=3, label=r'$I = \SI{0.5}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\volt}$')
plt.legend(loc='best')
plt.savefig('build/plot3.pdf')
plt.clf()

plt.plot(l4*1.8, I4, '.',markersize=3, label=r'$I = \SI{0.6}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\volt}$')
plt.legend(loc='best')
plt.savefig('build/plot4.pdf')
plt.clf()

plt.plot(l5*1.8, I5, '.',markersize=3, label=r'$I = \SI{0.7}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\volt}$')
plt.legend(loc='best')
plt.savefig('build/plot5.pdf')
plt.clf()

plt.plot(l6*1.8, I6, '.',markersize=3, label=r'$I = \SI{0.8}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\volt}$')
plt.legend(loc='best')
plt.savefig('build/plot6.pdf')
plt.clf()

plt.plot(l7*1.8, I7, '.',markersize=3, label=r'$I = \SI{0.9}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\volt}$')
plt.legend(loc='best')
plt.savefig('build/plot7.pdf')
plt.clf()

plt.plot(l8*1.8, I8, '.', markersize=3, label=r'$I = \SI{1.0}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\volt}$')
plt.legend(loc='best')
plt.savefig('build/plot8.pdf')
plt.clf()

print(len(l0)+len(l1)+len(l2)+len(l3)+len(l4)+len(l5)+len(l6)+len(l7)+len(l8))
