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

)
from error_calculation import(
    MeanError
)

from scipy.integrate import quad
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


def Temp(R):
    return 0.00134*R**2+2.296*R-243.02

def Wie(T,a,b,c):
    return -b/(2*a) + np.sqrt((b**2)/(4*a**2)+(c+T)/a)

def C_v(alpha, kappa, V_0, T, C_p):
    return C_p - (9 * alpha**2 * kappa * V_0 * T)

def debye_funktion(x):
    return (x**4 * np.exp(x))/(np.exp(x)-1)**2



def C_debye(T, theta):
    R = 8.3144598
    a = R*9*(T/theta)**3*list(map(lambda b: quad(debye_funktion, 0, b)[0], theta/T))
    return a

a = 0.00134
b = 2.296
c = 243.02

# 1celsius = 273.15K
#erstes t von 0.0256 bis 0.0299
#R = np.array([])
#auf ein ohm genau
#spannung eine nachkommastelle
#beim strom 2 nachkommastellen
#stefanie.roese@tu-dortmund.de
#cp-e1-140
t = np.array([0, 52+60*7 ,59+60*13 ,7+60*20 ,20+60*26 ,38+60*31 ,39+60*36 ,47+60*41 ,7+60*47 ,49+60*52 ,39+60*58 ,21+60*64 ,14+60*70 ,7+60*76 ,4+60*82 ,56+60*87 ,6+60*94 ,5+60*100 ,3+60*106 ,10+60*112 ,14+60*118 ,26+60*124 ])
u = np.array([13.9    ,16.5     ,16.6    ,16.7     ,19.4     ,19.5     ,19.5     ,19.6    ,19.6     ,19.6     ,19.6     ,19.6     ,19.6    ,19.6    ,19.6     ,19.6    ,19.6     ,19.6     ,19.6      ,19.6      ,19.6]) # +19.6
i = np.array([133.3   ,157      ,158.3   ,159      ,184      ,185      ,185.2    ,185.3   ,185.3    ,185.4    ,185.6    ,185.8    ,185.9   ,186     ,186.1    ,186.2   ,186.3    ,186.3    ,186.4     ,186.5     ,186.5]) # +186.5
i = i*10**(-3)
#T = Temp(R)
x = np.arange((80+10-273.15),(300-273.15+10),10)
T = np.arange(80+10,300+10,10)
alpha = np.array([9.75, 10.7, 11.5, 12.1, 12.65, 13.15, 13.6, 13.9, 14.25, 14.5, 14.75, 14.95, 15.2, 15.4, 15.6, 15.75, 15.9, 16.1, 16.25, 16.35, 16.5, 16.65]) # aus Anleitung
alpha = alpha**(-6)
alpha_interpol = np.diff(alpha)/2 + alpha[:-1]
T_interpol = np.diff(T)/2 + T[:-1]

m = 342/1000 # Masse kilgo
M = 63/1000 # kilogramm Pro Mol
n = m/M # Stoffmenge
kappa = 140 * 10**9 #Quelle: http://www.periodensystem-online.de/index.php?show=list&id=modify&prop=Kompressionsmodul&sel=oz&el=68
rho = 8920 # Quelle Wikipedia
V_0 = M / rho

E = u * i * np.diff(t) # zugeführte Energiemenge
C_m = E / (n * 10)

C_v = C_v(alpha_interpol, kappa, V_0, T_interpol, C_m)

integralwerte = np.array([ 3.3, 2.8, 2.7, 2.6, 1.9, 2.2, 2.0, 1.8])
theta_deb = integralwerte * T_interpol[0:8]

print(np.mean(theta_deb))
# d)

V = m/rho
N_L = 2.6867811*10**25 # quelle wikipedia
v_l = 4700
v_t = 2260

w_theo = ((18*np.pi**2 * N_L)/(V) * 1/ ( 1/v_l**3 + 2/v_t**3))**(1/3)
theta_theo = const.hbar * w_theo / const.k

# plot this shit
x = np.linspace(80,300)
plt.plot(x, C_debye(x, np.mean(theta_deb)))
plt.plot(x, C_debye(x, 345))
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.plot(T_interpol, C_v, 'x')
plt.axhline(y=3*const.R)


plt.savefig('build/fit.pdf')
#print( np.shape( 9*C_debye(x, np.mean(theta_deb)) ))
#print(const.R)
print(Wie(x,a,b,c)/1000)
