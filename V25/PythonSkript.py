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
l0 = np.array([4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.75, 5.8, 5.85, 5.9, 5.95, 6, 6.05, 6.06, 6.07, 6.1, 6.15, 6.2, 6.22, 6.24, 6.26, 6.28, 6.3, 6.32, 6.34, 6.36, 6.38, 6.4, 6.42, 6.44, 6.46, 6.48, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.5, 8.0, 9.0 ])*1.8
I0 = np.array([  0.06, 0.061, 0.061, 0.063, 0.065, 0.07, 0.072, 0.073, 0.076, 0.077, 0.079, 0.082, 0.091, 0.096, 0.102, 0.112, 0.13, 0.163, 0.193, 0.24, 0.293, 0.348, 0.424, 0.496, 0.571, 0.593, 0.607, 0.652, 0.733, 0.811, 0.843, 0.873, 0.901, 0.924, 0.949, 0.969, 0.983, 0.989, 0.998, 1.003, 1.002, 0.996, 0.986, 0.972, 0.951, 0.806, 0.625, 0.467, 0.329, 0.241, 0.187, 0.106, 0.087, 0.077 ])

# I=0.3 A , T=197°C

l1 = np.array([ 9.0, 8.0, 7.5, 7.0, 6.9, 6.8, 6.7, 6.68, 6.66, 6.64, 6.62, 6.6, 6.58, 6.56, 6.54, 6.52, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.88, 5.86, 5.84, 5.82, 5.8, 5.78, 5.76, 5.74, 5.72, 5.7, 5.68, 5.66, 5.64, 5.62, 5.6, 5.5, 5.4, 5.3, 5.2, 5.1, 5.0, 4.0, 3.0, 2.0 ])*1.8
I1 = np.array([ 0.084, 0.111, 0.154, 0.276, 0.333, 0.379, 0.423, 0.432, 0.434, 0.437, 0.442, 0.444, 0.448, 0.449, 0.448, 0.445, 0.443, 0.423, 0.401, 0.383, 0.381, 0.392, 0.423, 0.433, 0.438, 0.445, 0.453, 0.459, 0.468, 0.471, 0.480, 0.483, 0.489, 0.490, 0.489, 0.482, 0.479, 0.473, 0.423, 0.361, 0.304, 0.254, 0.219, 0.191, 0.087, 0.051, 0.040 ])

# I=0.4 A, T= 198°C

l2 = np.array([ 2.0, 3.0, 4.0, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.62, 5.64, 5.66, 5.68, 5.7, 5.72, 5.74, 5.76, 5.78, 5.8, 5.82, 5.84, 5.86, 5.88, 5.9, 5.92, 5.94, 5.96, 5.98, 6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.62, 6.64, 6.66, 6.68, 6.7, 6.72, 6.74, 6.76, 6.78, 6.8, 6.82, 6.84, 6.86, 6.88, 6.9, 6.92, 6.94, 6.96, 6.98, 7.0, 7.02, 7.04, 7.06, 7.08, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 9.0, 10.0 ])*1.8
I2 = np.array([ 0.041, 0.048, 0.081, 0.161, 0.172, 0.193, 0.216, 0.247, 0.296, 0.348, 0.361, 0.371, 0.382, 0.389, 0.398, 0.404, 0.413, 0.421, 0.424, 0.429, 0.431, 0.432, 0.434, 0.433, 0.429, 0.425, 0.424, 0.419, 0.412, 0.407, 0.367, 0.325, 0.284, 0.268, 0.280, 0.319, 0.330, 0.339, 0.347, 0.356, 0.368, 0.370, 0.375, 0.383, 0.388, 0.392, 0.396, 0.400, 0.408, 0.407, 0.408, 0.408, 0.412, 0.411, 0.411, 0.411, 0.409, 0.407, 0.402, 0.398, 0.393, 0.359, 0.324, 0.289, 0.251, 0.220, 0.197, 0.178, 0.164, 0.153, 0.089, 0.057 ])

# I=0.5 A, T=198°C

l3 = np.array([ 10.0, 9.0, 8.0, 7.9, 7.8, 7.7, 7.6, 7.5, 7.4, 7.3, 7.2, 7.1, 7.08, 7.06, 7.04, 7.02, 7.0, 6.98, 6.96, 6.94, 6.92, 6.9, 6.88, 6.86, 6.84, 6.82, 6.8, 6.78, 6.76, 6.74, 6.72, 6.7, 6.68, 6.66, 6.64, 6.62, 6.6, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7, 5.68, 5.66, 5.64, 5.62, 5.6, 5.58, 5.56, 5.54, 5.52, 5.5, 5.48, 5.46, 5.44, 5.42, 5.4, 5.3, 5.2, 5.1, 5.0, 4.0, 3.0, 2.0 ])*1.8
I3 = np.array([ 0.069, 0.086, 0.143, 0.161, 0.174, 0.190, 0.202, 0.214, 0.236, 0.258, 0.282, 0.311, 0.314, 0.324, 0.327, 0.329, 0.340, 0.346, 0.347, 0.353, 0.357, 0.356, 0.357, 0.363, 0.366, 0.365, 0.366, 0.363, 0.361, 0.359, 0.357, 0.354, 0.352, 0.348, 0.344, 0.340, 0.335, 0.306, 0.272, 0.230, 0.198, 0.181, 0.190, 0.227, 0.282, 0.336, 0.356, 0.363, 0.371, 0.377, 0.381, 0.384, 0.387, 0.390, 0.393, 0.393, 0.392, 0.390, 0.388, 0.384, 0.379, 0.351, 0.321, 0.289, 0.264, 0.115, 0.067, 0.044 ])

# I=0.6A, T = 198°C

l4 = np.array([ 2.0, 3.0, 4.0, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.52, 5.54, 5.56, 5.58, 5.6, 5.62, 5.64, 5.66, 5.68, 5.7, 5.72, 5.74, 5.76, 5.78, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.02, 7.04, 7.06, 7.08, 7.1, 7.12, 7.14, 7.16, 7.18, 7.2, 7.22, 7.24, 7.26, 7.28, 7.3, 7.32, 7.34, 7.36, 7.38, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 9.0, 10.0 ])*1.8
I4 = np.array([ 0.051, 0.064, 0.121, 0.211, 0.224, 0.243, 0.271, 0.299, 0.332, 0.335, 0.341, 0.346, 0.353, 0.356, 0.359, 0.366, 0.367, 0.367, 0.363, 0.363, 0.362, 0.358, 0.350, 0.346, 0.316, 0.279, 0.239, 0.199, 0.165, 0.149, 0.162, 0.201, 0.244, 0.282, 0.312, 0.330, 0.332, 0.339, 0.340, 0.336, 0.335, 0.335, 0.340, 0.341, 0.341, 0.339, 0.338, 0.334, 0.333, 0.332, 0.331, 0.329, 0.325, 0.321, 0.317, 0.312, 0.291, 0.267, 0.244, 0.226, 0.213, 0.197, 0.112, 0.075 ])

# I=0.7A, T=198°C

l5 = np.array([ 10.0, 9.0, 8.0, 7.0, 6.98, 6.96, 6.94, 6.92, 6.9, 6.88, 6.86, 6.84, 6.82, 6.8, 6.78, 6.76, 6.74, 6.72, 6.7, 6.6, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7, 5.6, 5.58, 5.56, 5.54, 5.52, 5.5, 5.48, 5.46, 5.44, 5.42, 5.4, 5.38, 5.36, 5.34, 5.32, 5.3, 5.28, 5.26, 5.24, 5.22, 5.2, 5.18, 5.16, 5.14, 5.12, 5.1, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.0, 3.0, 2.0 ])*1.8
I5 = np.array([ 0.078, 0.097, 0.164, 0.307, 0.309, 0.310, 0.311, 0.313, 0.311, 0.310, 0.309, 0.309, 0.305, 0.304, 0.306, 0.300, 0.294, 0.291, 0.284, 0.255, 0.224, 0.192, 0.161, 0.134, 0.122, 0.132, 0.156, 0.199, 0.242, 0.287, 0.296, 0.302, 0.313, 0.317, 0.324, 0.327, 0.330, 0.329, 0.333, 0.334, 0.332, 0.335, 0.334, 0.336, 0.331, 0.329, 0.325, 0.322, 0.321, 0.323, 0.319, 0.315, 0.311, 0.309, 0.306, 0.279, 0.260, 0.238, 0.217, 0.198, 0.180, 0.129, 0.071, 0.047 ])

# I=0.8A, T=198°C

l6 = np.array([ 2.0, 3.0, 4.0, 5.0, 5.1, 5.2, 5.22, 5.24, 5.26, 5.28, 5.3, 5.32, 5.34, 5.36, 5.38, 5.4, 5.42, 5.44, 5.46, 5.48, 5.5, 5.52, 5.54, 5.56, 5.58, 5.6, 5.62, 5.64, 5.66, 5.68, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.82, 6.84, 6.86, 6.88, 6.9, 6.92, 6.94, 6.96, 6.98, 7.0, 7.02, 7.04, 7.06, 7.08, 7.1, 7.12, 7.14, 7.16, 7.18, 7.2, 7.22, 7.24, 7.26, 7.28, 7.3, 7.32, 7.34, 7.36, 7.38, 7.4, 7.42, 7.44, 7.46, 7.48, 7.5, 7.52, 7.54, 7.56, 7.58, 7.6, 7.7, 7.8, 7.9, 8.0, 9.0, 10.0 ])*1.8
I6 = np.array([ 0.049, 0.066, 0.113, 0.226, 0.244, 0.263, 0.265, 0.267, 0.271, 0.274, 0.278, 0.284, 0.287, 0.289, 0.293, 0.298, 0.299, 0.300, 0.304, 0.309, 0.310, 0.312, 0.314, 0.312, 0.312, 0.311, 0.310, 0.308, 0.305, 0.303, 0.299, 0.268, 0.235, 0.203, 0.176, 0.146, 0.123, 0.116, 0.125, 0.154, 0.185, 0.217, 0.221, 0.228, 0.235, 0.241, 0.243, 0.250, 0.248, 0.254, 0.258, 0.261, 0.266, 0.269, 0.273, 0.276, 0.279, 0.281, 0.282, 0.287, 0.291, 0.291, 0.292, 0.293, 0.291, 0.291, 0.291, 0.293, 0.292, 0.291, 0.295, 0.291, 0.290, 0.289, 0.287, 0.285, 0.281, 0.279, 0.276, 0.273, 0.271, 0.269, 0.258, 0.243, 0.229, 0.216, 0.214, 0.088 ])

#I=0.9A, T=198°C
l7 = np.array([ 10.0, 9.0, 8.0, 7.9, 7.8, 7.7, 7.6, 7.5, 7.4, 7.38, 7.36, 7.34, 7.32, 7.3, 7.28, 7.26, 7.24, 7.22, 7.2, 7.18, 7.16, 7.14, 7.12, 7.1, 7.08, 7.06, 7.04, 7.02, 7.0, 6.98, 6.96, 6.94, 6.92, 6.9, 6.8, 6.7, 6.6, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7, 5.6, 5.58, 5.56, 5.54, 5.52, 5.5, 5.48, 5.46, 5.44, 5.42, 5.4, 5.38, 5.36, 5.34, 5.32, 5.3, 5.28, 5.26, 5.24, 5.22, 5.2, 5.18, 5.16, 5.14, 5.12, 5.1, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.0, 3.0, 2.0 ])*1.8
I7 = np.array([ 0.089, 0.109, 0.183, 0.198, 0.208, 0.221, 0.232, 0.246, 0.259, 0.261, 0.262, 0.263, 0.269, 0.270, 0.272, 0.274, 0.275, 0.276, 0.278, 0.280, 0.281, 0.282, 0.282, 0.284, 0.284, 0.284, 0.284, 0.284, 0.281, 0.282, 0.279, 0.278, 0.275, 0.273, 0.259, 0.235, 0.211, 0.182, 0.156, 0.131, 0.116, 0.103, 0.107, 0.123, 0.149, 0.184, 0.224, 0.232, 0.241, 0.248, 0.255, 0.260, 0.267, 0.273, 0.278, 0.280, 0.284, 0.288, 0.291, 0.293, 0.296, 0.298, 0.299, 0.299, 0.303, 0.300, 0.299, 0.298, 0.296, 0.295, 0.294, 0.292, 0.281, 0.265, 0.248, 0.233, 0.221, 0.205, 0.149, 0.087, 0.061 ])

#I=1.0A, T=198°C
l8 = np.array([ 2.0, 3.0, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.12, 5.14, 5.16, 5.18, 5.2, 5.22, 5.24, 5.26, 5.28, 5.3, 5.32, 5.34, 5.36, 5.38, 5.4, 5.42, 5.44, 5.46, 5.48, 5.5, 5.52, 5.54, 5.56, 5.58, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.22, 7.24, 7.26, 7.28, 7.3, 7.32, 7.34, 7.36, 7.38, 7.4, 7.42, 7.44, 7.46, 7.48, 7.5, 7.52, 7.54, 7.56, 7.58, 7.6, 7.7, 7.8, 7.9, 8.0, 9.0, 10.0 ])*1.8
I8 = np.array([ 0.061, 0.076, 0.127, 0.134, 0.141, 0.149, 0.160, 0.172, 0.185, 0.199, 0.213, 0.226, 0.237, 0.251, 0.253, 0.253, 0.254, 0.256, 0.259, 0.261, 0.263, 0.265, 0.268, 0.271, 0.272, 0.275, 0.279, 0.280, 0.281, 0.282, 0.285, 0.284, 0.284, 0.284, 0.283, 0.281, 0.277, 0.276, 0.272, 0.250, 0.219, 0.187, 0.159, 0.137, 0.117, 0.101, 0.096, 0.102, 0.121, 0.147, 0.173, 0.203, 0.223, 0.241, 0.258, 0.259, 0.261, 0.265, 0.267, 0.272, 0.273, 0.274, 0.274, 0.274, 0.276, 0.276, 0.275, 0.274, 0.273, 0.274, 0.270, 0.268, 0.265, 0.261, 0.261, 0.252, 0.240, 0.227, 0.216, 0.130, 0.091 ])


mu_0 = const.mu_0
#mu_r = 1-6.4*10**(-6) # <---- Wieso?! Bisschen klein für einen Magneten, nicht?
mu_r = 8000
a    = 2.5*10**(-3)
r    = 1.3*a #vielleicht
l = 0.445 #meter
L = 7 * 10 ** (-2)


#def B(I_b):
#    return mu_0*mu_r*I_b/np.pi * a/(r**2)

def dB(B_tmp):
    return 0.968*B_tmp/a

def s(m):
    return m*x

def gauss(x,mu,sigma,b,a):
    return a*np.exp(-1/2 * ((x-mu)/sigma)**2)+b

def doppel_gauss(x, mu, mu2, sigma, a, sigma2, a2, b):
    return a*np.exp(-1/2 * ((x-mu)/sigma)**2) + a2*np.exp(-1/2 * ((x-mu2)/sigma2)**2)+b

def linear_fit(x, m_fit, b_fit):
    return m_fit*x + b_fit

params0, cov0 = curve_fit(gauss,l0,I0, p0=[11.522,0.618,0,1])
params1, cov1 = curve_fit(doppel_gauss,l1,I1, p0=[11.522,11.522,0.618,1,0.618,1,0])
params2, cov2 = curve_fit(doppel_gauss,l2,I2, p0=[10.8,12.8,0.6,1,0.6,1,0])
params3, cov3 = curve_fit(doppel_gauss,l3,I3, p0=[10,12.5,0.8,1,0.8,1,0])
params4, cov4 = curve_fit(doppel_gauss,l4,I4, p0=[10,13,0.8,1,0.8,1,0])
params5, cov5 = curve_fit(doppel_gauss,l5,I5, p0=[9.3,12.5,0.8,1,0.8,1,0])
params6, cov6 = curve_fit(doppel_gauss,l6,I6, p0=[10,13,0.5,1,0.5,1,0])
params7, cov7 = curve_fit(doppel_gauss,l7,I7, p0=[9.3,12.3,0.8,1,0.8,1,0])
params8, cov8 = curve_fit(doppel_gauss,l8,I8, p0=[9.8,13.3,0.8,1,0.8,1,0])
print(params6)

s_mitte = ufloat(params0[0], np.sqrt(cov0[0,0]))
write('build/s_mitte.tex', make_SI(s_mitte, r'\milli\metre', figures=1))

#Maxima von Hand
s0_1 = 6.56 * 1.8
s1_1 = 5.68 * 1.8
s1_l_h = abs(min(s0_1, s1_1) - s_mitte)
s1_r_h = abs(max(s0_1, s1_1) - s_mitte)

s0_2 = 5.86 * 1.8
s1_2 = 6.98 * 1.8
s2_l_h = abs(min(s0_2, s1_2) - s_mitte)
s2_r_h = abs(max(s0_2, s1_2) - s_mitte)

s0_3 = 6.82 * 1.8
s1_3 = 5.51 * 1.8
s3_l_h = abs(min(s0_3, s1_3) - s_mitte)
s3_r_h = abs(max(s0_3, s1_3) - s_mitte)

s0_4 = 5.69 * 1.8
s1_4 = 7.17 * 1.8
s4_l_h = abs(min(s0_4, s1_4) - s_mitte)
s4_r_h = abs(max(s0_4, s1_4) - s_mitte)

s0_5 = 6.92 * 1.8
s1_5 = 5.38 * 1.8
s5_l_h = abs(min(s0_5, s1_5) - s_mitte)
s5_r_h = abs(max(s0_5, s1_5) - s_mitte)

s0_6 = 5.57 * 1.8
s1_6 = 7.34 * 1.8
s6_l_h = abs(min(s0_6, s1_6) - s_mitte)
s6_r_h = abs(max(s0_6, s1_6) - s_mitte)

s0_7 = 7.08 * 1.8
s1_7 = 5.25 * 1.8
s7_l_h = abs(min(s0_7, s1_7) - s_mitte)
s7_r_h = abs(max(s0_7, s1_7) - s_mitte)

s0_8 = 5.47 * 1.8
s1_8 = 7.42 * 1.8
s8_l_h = abs(min(s0_8, s1_8) - s_mitte)
s8_r_h = abs(max(s0_8, s1_8) - s_mitte)

s_g_l_h_n = np.array([s2_l_h.n, s4_l_h.n, s6_l_h.n, s8_l_h.n]) * 10**(-3) # s_g_l_h_n = S-werte, die Gerade sind, die für die Linken Maxima stehen, die per Hand abgelesen wurden, und davon die Nominellen Werte
s_g_r_h_n = np.array([s2_r_h.n, s4_r_h.n, s6_r_h.n, s8_r_h.n]) * 10**(-3)
s_g_l_h_s = np.array([s2_l_h.s, s4_l_h.s, s6_l_h.s, s8_l_h.s]) * 10**(-3)
s_g_r_h_s = np.array([s2_r_h.s, s4_r_h.s, s6_r_h.s, s8_r_h.s]) * 10**(-3)

s_u_l_h_n = np.array([s1_l_h.n ,s3_l_h.n, s5_l_h.n, s7_l_h.n]) * 10**(-3)
s_u_r_h_n = np.array([s1_r_h.n ,s3_r_h.n, s5_r_h.n, s7_r_h.n]) * 10**(-3)
s_u_l_h_s = np.array([s1_l_h.s ,s3_l_h.s, s5_l_h.s, s7_l_h.s]) * 10**(-3)
s_u_r_h_s = np.array([s1_r_h.s ,s3_r_h.s, s5_r_h.s, s7_r_h.s]) * 10**(-3)

x_space_0 = np.linspace(np.amin(l0),np.amax(l0),1000)
plt.plot(x_space_0,gauss(x_space_0, *params0), linewidth=1)
plt.plot(l0, I0, '.', markersize=3,label=r'$I = \SI{0}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.ylabel(r'$U / \si{\volt}$')
plt.legend(loc='best')
plt.savefig('build/plot0.pdf')
plt.clf()


x_space_1 = np.linspace(np.amin(l1),np.amax(l1),1000)
plt.plot(x_space_1,doppel_gauss(x_space_1, *params1), linewidth=1)
plt.plot(l1, I1, '.',markersize=3, label=r'$I = \SI{0.3}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.ylabel(r'$U / \si{\volt}$')
plt.axvline(x=s0_1,linewidth=1, color='g', linestyle='dashed')
plt.axvline(x=s1_1,linewidth=1, color='g', linestyle='dashed')
plt.legend(loc='best')
plt.savefig('build/plot1.pdf')
plt.clf()


x_space_2 = np.linspace(np.amin(l2),np.amax(l2),1000)
plt.plot(x_space_2,doppel_gauss(x_space_2, *params2), linewidth=1)
plt.plot(l2, I2, '.',markersize=3, label=r'$I = \SI{0.4}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.ylabel(r'$U / \si{\volt}$')
plt.axvline(x=s0_2,linewidth=1, color='g', linestyle='dashed')
plt.axvline(x=s1_2,linewidth=1, color='g', linestyle='dashed')
plt.legend(loc='best')
plt.savefig('build/plot2.pdf')
plt.clf()

x_space_3 = np.linspace(np.amin(l3),np.amax(l3),1000)
plt.plot(x_space_3,doppel_gauss(x_space_3, *params3), linewidth=1)
plt.plot(l3, I3, '.',markersize=3, label=r'$I = \SI{0.5}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.ylabel(r'$U / \si{\volt}$')
plt.axvline(x=s0_3,linewidth=1, color='g', linestyle='dashed')
plt.axvline(x=s1_3,linewidth=1, color='g', linestyle='dashed')
plt.legend(loc='best')
plt.savefig('build/plot3.pdf')
plt.clf()

x_space_4 = np.linspace(np.amin(l4),np.amax(l4),1000)
plt.plot(x_space_4,doppel_gauss(x_space_4, *params4), linewidth=1)
plt.plot(l4, I4, '.',markersize=3, label=r'$I = \SI{0.6}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.ylabel(r'$U / \si{\volt}$')
plt.axvline(x=s0_4,linewidth=1, color='g', linestyle='dashed')
plt.axvline(x=s1_4,linewidth=1, color='g', linestyle='dashed')
plt.legend(loc='best')
plt.savefig('build/plot4.pdf')
plt.clf()

x_space_5 = np.linspace(np.amin(l5),np.amax(l5),1000)
plt.plot(x_space_5,doppel_gauss(x_space_5, *params5), linewidth=1)
plt.plot(l5, I5, '.',markersize=3, label=r'$I = \SI{0.7}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.ylabel(r'$U / \si{\volt}$')
plt.axvline(x=s0_5,linewidth=1, color='g', linestyle='dashed')
plt.axvline(x=s1_5,linewidth=1, color='g', linestyle='dashed')
plt.legend(loc='best')
plt.savefig('build/plot5.pdf')
plt.clf()

x_space_6 = np.linspace(np.amin(l6),np.amax(l6),1000)
plt.plot(x_space_6,doppel_gauss(x_space_6, *params6), linewidth=1)
plt.plot(l6, I6, '.',markersize=3, label=r'$I = \SI{0.8}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.ylabel(r'$U / \si{\volt}$')
plt.axvline(x=s0_6,linewidth=1, color='g', linestyle='dashed')
plt.axvline(x=s1_6,linewidth=1, color='g', linestyle='dashed')
plt.legend(loc='best')
plt.savefig('build/plot6.pdf')
plt.clf()

x_space_7 = np.linspace(np.amin(l7),np.amax(l7),1000)
plt.plot(x_space_7,doppel_gauss(x_space_7, *params7), linewidth=1)
plt.plot(l7, I7, '.',markersize=3, label=r'$I = \SI{0.9}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.ylabel(r'$U / \si{\volt}$')
plt.axvline(x=s0_7,linewidth=1, color='g', linestyle='dashed')
plt.axvline(x=s1_7,linewidth=1, color='g', linestyle='dashed')
plt.legend(loc='best')
plt.savefig('build/plot7.pdf')
plt.clf()

x_space_8 = np.linspace(np.amin(l8),np.amax(l8),1000)
plt.plot(x_space_8,doppel_gauss(x_space_8, *params8), linewidth=1)
plt.plot(l8, I8, '.', markersize=3, label=r'$I = \SI{1.0}{\ampere}$') # so richtig mit der 1.8? oder */10 ?
plt.xlabel(r'$l / \si{\milli\metre}$')
plt.ylabel(r'$U / \si{\volt}$')
plt.axvline(x=s0_8,linewidth=1,color='g', linestyle='dashed')
plt.axvline(x=s1_8,linewidth=1,color='g', linestyle='dashed')
plt.legend(loc='best')
plt.savefig('build/plot8.pdf')
plt.clf()

#  Bestimme Stuff
s1_l = abs(min( ufloat(params1[0], np.sqrt(cov1[0,0])), ufloat(params1[1], np.sqrt(cov1[1,1])) ) - s_mitte)
s2_l = abs(min( ufloat(params2[0], np.sqrt(cov2[0,0])), ufloat(params2[1], np.sqrt(cov2[1,1])) ) - s_mitte)
s3_l = abs(min( ufloat(params3[0], np.sqrt(cov3[0,0])), ufloat(params3[1], np.sqrt(cov3[1,1])) ) - s_mitte)
s4_l = abs(min( ufloat(params4[0], np.sqrt(cov4[0,0])), ufloat(params4[1], np.sqrt(cov4[1,1])) ) - s_mitte)
s5_l = abs(min( ufloat(params5[0], np.sqrt(cov5[0,0])), ufloat(params5[1], np.sqrt(cov5[1,1])) ) - s_mitte)
s6_l = abs(min( ufloat(params6[0], np.sqrt(cov6[0,0])), ufloat(params6[1], np.sqrt(cov6[1,1])) ) - s_mitte)
s7_l = abs(min( ufloat(params7[0], np.sqrt(cov7[0,0])), ufloat(params7[1], np.sqrt(cov7[1,1])) ) - s_mitte)
s8_l = abs(min( ufloat(params8[0], np.sqrt(cov8[0,0])), ufloat(params8[1], np.sqrt(cov8[1,1])) ) - s_mitte)
mu1_ges = np.array([min(params1[0],params1[1]),min(params2[0],params2[1]),min(params3[0],params3[1]),min(params4[0],params4[1]),min(params5[0],params5[1]),min(params6[0],params6[1]),min(params7[0],params7[1]),min(params8[0],params8[1]])

s1_r = abs(max( ufloat(params1[0], np.sqrt(cov1[0,0])), ufloat(params1[1], np.sqrt(cov1[1,1])) ) - s_mitte)
s2_r = abs(max( ufloat(params2[0], np.sqrt(cov2[0,0])), ufloat(params2[1], np.sqrt(cov2[1,1])) ) - s_mitte)
s3_r = abs(max( ufloat(params3[0], np.sqrt(cov3[0,0])), ufloat(params3[1], np.sqrt(cov3[1,1])) ) - s_mitte)
s4_r = abs(max( ufloat(params4[0], np.sqrt(cov4[0,0])), ufloat(params4[1], np.sqrt(cov4[1,1])) ) - s_mitte)
s5_r = abs(max( ufloat(params5[0], np.sqrt(cov5[0,0])), ufloat(params5[1], np.sqrt(cov5[1,1])) ) - s_mitte)
s6_r = abs(max( ufloat(params6[0], np.sqrt(cov6[0,0])), ufloat(params6[1], np.sqrt(cov6[1,1])) ) - s_mitte)
s7_r = abs(max( ufloat(params7[0], np.sqrt(cov7[0,0])), ufloat(params7[1], np.sqrt(cov7[1,1])) ) - s_mitte)
s8_r = abs(max( ufloat(params8[0], np.sqrt(cov8[0,0])), ufloat(params8[1], np.sqrt(cov8[1,1])) ) - s_mitte)
mu2_ges = np.array([max(params1[0],params1[1]),max(params2[0],params2[1]),max(params3[0],params3[1]),max(params4[0],params4[1]),max(params5[0],params5[1]),max(params6[0],params6[1]),max(params7[0],params7[1]),max(params8[0],params8[1])])

s_g_l_n = np.array([s2_l.n, s4_l.n, s6_l.n, s8_l.n]) *10**(-3) # jetzt in Meter
s_g_r_n = np.array([s2_r.n, s4_r.n, s6_r.n, s8_r.n]) *10**(-3) # jetzt in Meter
s_g_l_s = np.array([s2_l.s, s4_l.s, s6_l.s, s8_l.s]) *10**(-3) # jetzt in Meter
s_g_r_s = np.array([s2_r.s, s4_r.s, s6_r.s, s8_r.s]) *10**(-3) # jetzt in Meter

s_u_l_n = np.array([s1_l.n, s3_l.n, s5_l.n, s7_l.n]) *10**(-3) # jetzt in Meter
s_u_r_n = np.array([s1_r.n, s3_r.n, s5_r.n, s7_r.n]) *10**(-3) # jetzt in Meter
s_u_l_s = np.array([s1_l.s, s3_l.s, s5_l.s, s7_l.s]) *10**(-3) # jetzt in Meter
s_u_r_s = np.array([s1_r.s, s3_r.s, s5_r.s, s7_r.s]) *10**(-3) # jetzt in Meter

s_l_ges = np.array([s1_l.n,s2_l.n,s3_l.n,s4_l.n,s5_l.n,s6_l.n,s7_l.n,s8_l.n])
s_r_ges = np.array([s1_r.n,s2_r.n,s3_r.n,s4_r.n,s5_r.n,s6_r.n,s7_r.n,s8_r.n])


T_g = np.array([198, 198, 198, 198]) + 273.15 # Celsius oder Ke(l)vin?!
T_u = np.array([197, 198, 198, 198]) + 273.15 # Celsius oder Ke(l)vin?!
I = np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

B_abgelesen_g = np.array([0.3, 0.45, 0.595, 0.72 ]) # von 0.3 bis 1 die abgelesenen werte
B_abgelesen_u = np.array([0.22, 0.38, 0.525, 0.66]) # von 0.3 bis 1 die abgelesenen werte
B_ges = np.array(([0.22, 0.3, 0.38, 0.45, 0.525, 0.595, 0.66, 0.72]))

Vorf_g = l * L * (1-L/(2*l))/(6*const.k * T_g) * dB(B_abgelesen_g)
Vorf_u = l * L * (1-L/(2*l))/(6*const.k * T_u) * dB(B_abgelesen_u)


params9_g, cov9_g = curve_fit(linear_fit,Vorf_g,s_g_l_n, p0=[9*10**(a,b,c,d,e-24), 0], sigma=s_g_l_s)
params10_g, cov10_g = curve_fit(linear_fit,Vorf_g,s_g_r_n, p0=[9*10**(-24), 0], sigma=s_g_r_s)
params11_g, cov11_g = curve_fit(linear_fit,Vorf_g,s_g_l_h_n, p0=[9*10**(-24), 0], sigma=s_g_l_h_s)
params12_g, cov12_g = curve_fit(linear_fit,Vorf_g,s_g_r_h_n, p0=[9*10**(-24), 0], sigma=s_g_r_h_s)

params9_u, cov9_u = curve_fit(linear_fit,Vorf_u,s_u_l_n, p0=[9*10**(-24), 0], sigma=s_u_l_s)
params10_u, cov10_u = curve_fit(linear_fit,Vorf_u,s_u_r_n, p0=[9*10**(-24), 0], sigma=s_u_r_s)
params11_u, cov11_u = curve_fit(linear_fit,Vorf_u,s_u_l_h_n, p0=[9*10**(-24), 0], sigma=s_u_l_h_s)
params12_u, cov12_u = curve_fit(linear_fit,Vorf_u,s_u_r_h_n, p0=[9*10**(-24), 0], sigma=s_u_r_h_s)

x_space_9 = np.linspace(np.amin(Vorf_u),np.amax(Vorf_g),1000)
plt.plot(x_space_9,linear_fit(x_space_9, *params9_g)*10**3, 'r-',linewidth=1, label=r'Abstände mit Fits - Linke Maxima')
plt.plot(x_space_9,linear_fit(x_space_9, *params11_g)*10**3, 'b-', linewidth=1, label=r'Abstände per Hand - Linke Maxima')
plt.plot(x_space_9,linear_fit(x_space_9, *params9_u)*10**3, 'r-',linewidth=1)
plt.plot(x_space_9,linear_fit(x_space_9, *params11_u)*10**3, 'b-', linewidth=1)

plt.errorbar(Vorf_g, s_g_l_n*10**3,yerr=s_g_l_s*10**3, fmt='r.')
plt.errorbar(Vorf_u, s_u_l_n*10**3,yerr=s_u_l_s*10**3, fmt='r.')

plt.errorbar(Vorf_g, s_g_l_h_n*10**3,yerr=s_g_l_h_s*10**3, fmt='b.')
plt.errorbar(Vorf_u, s_u_l_h_n*10**3,yerr=s_u_l_h_s*10**3, fmt='b.')

plt.ylabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\tesla\metre\per\joule}$')
plt.legend(loc='best')
plt.savefig('build/plot_links.pdf')
plt.clf()

plt.plot(x_space_9,linear_fit(x_space_9, *params10_g)*10**3, 'r-',linewidth=1, label=r'Abstände mit Fits - Rechts Maxima')
plt.plot(x_space_9,linear_fit(x_space_9, *params12_g)*10**3, 'b-', linewidth=1, label=r'Abstände per Hand abgelesen - Rechts Maxima')
plt.plot(x_space_9,linear_fit(x_space_9, *params10_u)*10**3, 'r-',linewidth=1)
plt.plot(x_space_9,linear_fit(x_space_9, *params12_u)*10**3, 'b-', linewidth=1)

plt.errorbar(Vorf_g, s_g_r_n*10**3,yerr=s_g_r_s*10**3, fmt='r.')
plt.errorbar(Vorf_u, s_u_r_n*10**3,yerr=s_u_r_s*10**3, fmt='r.')

plt.errorbar(Vorf_g, s_g_r_h_n*10**3,yerr=s_g_r_h_s*10**3, fmt='b.')
plt.errorbar(Vorf_u, s_u_r_h_n*10**3,yerr=s_u_r_h_s*10**3, fmt='b.')
plt.ylabel(r'$l / \si{\milli\metre}$')
plt.xlabel(r'$U / \si{\tesla\metre\per\joule}$')
plt.legend(loc='best')
plt.savefig('build/plot_rechts.pdf')
plt.clf()

### Das sollte im Bestenfall direkt das Bohrsche Magneton ergeben, da mu_sz = 0.5*2 * mu_b
print("Per fit")
print(params9_g[0])
print(params9_u[0])
print(params10_g[0])
print(params10_u[0])
print("Per Hand")
print(params11_g[0])
print(params11_u[0])
print(params12_g[0])
print(params12_u[0])


## Tabellenstuff

write('build/Tabelle_a.tex', make_table([I, B_ges, mu1_ges, mu2_ges, s_l_ges, s_r_ges ],[1, 1, 1, 1, 1,1]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_a_texformat.tex', make_full_table(
     'Messdaten Kapazitätsmessbrücke.',
     'table:A2',
     'build/Tabelle_a.tex',
     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                               # die Multicolumns sein sollen
     [r'$I  \:/\:  \si{ampere}$',
     r'$B \:/\: \si{\tesla}$',
     r'$\mu_1 \:/\: \si{\metre}$',
     r'$\mu_2 \:/\: \si{\metre}$',
     r'$s_1 \:/\: \si{\metre}$',
     r'$s_2 \:/\: \si{\metre}$']))
