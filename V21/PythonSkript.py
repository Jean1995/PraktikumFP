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

from scipy.integrate import quad
from scipy.optimize import curve_fit
################################################ Finish importing custom libraries #################################################







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
## hard coded limits
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




plt.style.use('seaborn-darkgrid')
plt.set_cmap('Set2')

### Funktionen

def B(I, N, R):
    return const.mu_0 * (8*I*N)/(np.sqrt(125)*R)

def f(x, m, b):
    return m*x+b

def KSpin(J, g_F):
    g_j = 2.00232
    return J * (g_j/g_F - 1)

def zee_linear(B_tmp, g_tmp):
    mu_b = const.e * const.hbar / (2 * const.m_e)
    return g_tmp * B_tmp * mu_b

def zee_quad(B_tmp, g_tmp, E):
    mu_b = const.e * const.hbar / (2 * const.m_e)
    return g_tmp**2 * mu_b**2 * B_tmp**2 * (1-2*1)/E



### Erdmagnetfeld

I_vert = ufloat(2.43 * 0.1, 0.01*0.1) # 1 Umdrehung =: 0.1 A, Ablesefehler 0.01 Umdrehungen
B_vert = B(I_vert, 20, 0.11735)

write('build/I_vert.tex', make_SI(I_vert, r'\ampere', figures=1))
write('build/B_vert.tex', make_SI(B_vert*10**6, r'\micro\tesla', figures=1))

### Resonanzfrequenzen

Umd_1_1_n = np.array([5.66, 6.8, 4.12, 3.52, 2.06, 1.0, 0.65, 2.54, 4.22, 3.4]) # 1. Resonanz Sweep
Umd_1_2_n = np.array([0, 0.04, 0.15, 0.22, 0.31, 0.38, 0.44, 0.46, 0.47, 0.55]) # 1. Resonanz fest
Umd_2_1_n = np.array([6.88, 9.12, 7.65, 8.25, 8.03, 8.1, 8.91, 5.88, 6.22, 5.57]) # 2. Resonanz sweep
Umd_2_2_n = np.array([0, 0.04, 0.15, 0.22, 0.31, 0.38, 0.44, 0.6, 0.67, 0.77]) # 2. Resonanz fest
ablesefehler = Umd_1_1_n*0 + 0.01


Umd_1_1 = unp.uarray(Umd_1_1_n, ablesefehler)
Umd_1_2 = unp.uarray(Umd_1_2_n, ablesefehler)
Umd_2_1 = unp.uarray(Umd_2_1_n, ablesefehler)
Umd_2_2 = unp.uarray(Umd_2_2_n, ablesefehler)

I_1_1 = Umd_1_1*0.1 # 1 Umdrehung = 0.1A bei Sweepspule
I_1_2 = Umd_1_2*0.3 # 1 Umdrehung = 0.3A bei Horizontalspule
I_2_1 = Umd_2_1*0.1
I_2_2 = Umd_2_2*0.3

B_1_1 = B(I_1_1, 11, 0.1639)
B_1_2 = B(I_1_2, 154, 0.1549)
B_2_1 = B(I_2_1, 11, 0.1639)
B_2_2 = B(I_2_2, 154, 0.1549)

B_1_ges = B_1_1 + B_1_2 # Gesmates B-Feld 1. Resonanz
B_2_ges = B_2_1 + B_2_2 # Gesates B-Feld 2. Resonanz

nu = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])*1000 # Frequenzen

write('build/messwerte.tex', make_table([nu/1000, unp.nominal_values(I_1_1), unp.nominal_values(I_1_2), unp.nominal_values(I_2_1), unp.nominal_values(I_2_2), B_1_ges*10**6, B_2_ges*10**6],[0,3,3,3,3,2,2,2,2]))

### Bestimmung Lande Faktor aus Fit

params_1, cov_1= curve_fit(f, nu, unp.nominal_values(B_1_ges))
params_2, cov_2 = curve_fit(f, nu, unp.nominal_values(B_2_ges))
error_1 = np.sqrt(np.diag(cov_1))
error_2 = np.sqrt(np.diag(cov_2))

nu_plot = np.linspace(100, 1000, 1000)*1000

### 1. Resonanz

plt.errorbar(nu/1000, unp.nominal_values(B_1_ges)*10**6, fmt=',', yerr=unp.std_devs(B_1_ges)*10**6, label='Messdaten')
plt.plot(nu_plot/1000, f(nu_plot, params_1[0], params_1[1])*10**6, label=r'Linearer Fit')
plt.legend(loc="best")
plt.xlabel(r'$\nu \:/\: \si{\kilo\hertz}$')
plt.ylabel(r'$B \:/\: \si{\micro\tesla}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/fit_1.pdf')
print(params_1)

m_1 = ufloat(params_1[0], error_1[0])
b_1 = ufloat(params_1[1], error_1[1])

write('build/m_1.tex', make_SI(m_1*10**9, r'\nano\tesla\second', figures=1))
write('build/b_1.tex', make_SI(b_1*10**6, r'\micro\tesla', figures=1))

g_1 = 4*np.pi * const.m_e / (const.e * m_1)
write('build/g_1.tex', make_SI(g_1, r'', figures=1))

plt.clf()

### 2. Resonanz

plt.errorbar(nu/1000, unp.nominal_values(B_2_ges)*10**6, fmt=',', yerr=unp.std_devs(B_2_ges)*10**6, label='Messdaten')
plt.plot(nu_plot/1000, f(nu_plot, params_2[0], params_2[1])*10**6, label=r'Linearer Fit')
plt.legend(loc="best")
plt.xlabel(r'$\nu \:/\: \si{\kilo\hertz}$')
plt.ylabel(r'$B \:/\: \si{\micro\tesla}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/fit_2.pdf')
print(params_2)

m_2 = ufloat(params_2[0], error_2[0])
b_2 = ufloat(params_2[1], error_2[1])

write('build/m_2.tex', make_SI(m_2*10**9, r'\nano\tesla\second', figures=1))
write('build/b_2.tex', make_SI(b_2*10**6, r'\micro\tesla', figures=1))

g_2 = 4*np.pi * const.m_e / (const.e * m_2)
write('build/g_2.tex', make_SI(g_2, r'', figures=1))

plt.clf()

### Kernspin

I_1 = KSpin(0.5, g_1)
I_2 = KSpin(0.5, g_2) # Formel muss noch hergeleitet werden, habe ich so angenommen
write('build/I_1.tex', make_SI(I_1, r'', figures=1))
write('build/I_2.tex', make_SI(I_2, r'', figures=1)) #hat das zeug ne Einheit? eigentlich schon, oder?

### Amplitudenverhältnis

A1 = 64 # linke Amplitude, Pixelhöhe
A2 = 98 # rechte Amplitude, Pixelhöhe
write('build/A_1.tex', make_SI(A1, r'', figures=1))
write('build/A_2.tex', make_SI(A2, r'', figures=1))
A_V = A2/A1
write('build/A_V.tex', make_SI(A_V, r'', figures=1))

Rb_85 = 72.17
Rb_87 = 27.83
write('build/Rb_85.tex', make_SI(Rb_85, r'\percent', figures=2))
write('build/Rb_87.tex', make_SI(Rb_87, r'\percent', figures=2))
Rb_V = Rb_85/Rb_87
write('build/Rb_V.tex', make_SI(Rb_V, r'', figures=1))

#Quadraaatischer Zeeman-Effekt

B_max = 250*10**(-6) # hochgerundeter maximaler Wert
g_f = 0.5 # hochgerundeter g_f Wert
E_1 = 4.53 * 10**(-24)
E_2 = 2.01 * 10**(-24)

lin_1 = zee_linear(B_max, g_f)
lin_2 = lin_1
quad_1 = zee_quad(B_max, g_f, E_1)
quad_2 = zee_quad(B_max, g_f, E_2)
write('build/zeeman_1.tex', make_SI(np.abs(lin_1/quad_1), r'', figures=0))
write('build/zeeman_2.tex', make_SI(np.abs(lin_2/quad_2), r'', figures=0))
