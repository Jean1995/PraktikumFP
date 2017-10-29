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

from uncertainties.unumpy import exp

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


### MESSDATEN

data = np.array([       0,   0,   0, 244, 324,  86,  87,  92,  71,  84,  54,  83,  67,  71,  94,  92,  69,  88,  73,  62,  67,  79,  92,  64,  74,  68,  68,  76,  54,  75,  72,  64,  64,  58,  55,  68,  64,  64,  73,  62,  49,  48,  57,  60,  46,  48,  62,  44,  42,  54,  50,  50,  48,  52,  41,  52,  40,  55,  40,  46,  52,  50,  52,  44,  51,  33,  40,  67,  51,  43,  38,  41,  36,  39,  52,  37,  34,  47,  49,  42,  66,  30,  39,  44,  47,  41,  31,  36,  30,  28,  43,  35,  41,  30,  31,  44,  28,  19,  36,  30,  27,  43,  39,  41,  28,  29,  26,  23,  32,  13,  30,  35,  24,  32,  21,  23,  18,  21,  27,  23,  23,  24,  27,  27,  24,  21,  27,  28,  32,  24,  26,  29,  23,  22,  21,  21,  24,  19,  26,  25,  16,  18,  27,  17,  19,  16,  21,  21,  18,  14,  14,  14,  19,  15,  20,  22,  16,   9,  18,  20,  29,  14,  15,  16,  19,  23,  17,  17,  10,  19,  14,  12,  16,  20,  16,  15,  15,  17,  18,  13,  12,  15,  10,  14,  11,  11,  16,  13,   9,   5,  14,  14,  18,  13,   9,  14,  18,   3,  14,   3,  15,  12,  12,  16,   3,   9,  17,   5,   7,  14,  11,   7,  15,  13,  15,  11,  10,  10,  11,  10,   7,  11,   7,  12,   9,   7,   7,  10,   5,   5,   8,   8,   6,  11,  10,  10,   9,  18,  11,   9,   6,  15,   9,   5,   9,  10,   6,   7,   8,   6,  11,   9,  10,   6,  11,   8,   5,   8,   9,  12,   6,   4,   5,   5,   5,   6,  10,  12,   4,   7,   9,   6,   5,  11,   7,   7,   6,   4,   2,  10,   8,   3,   7,   9,   3,   4,   6,   4,   6,   7,   4,   6,   8,   8,   7,   4,   3,   2,   6,   4,   8,   5,   3,   3,   5,   7,   2,   5,   6,   2,   5,   8,   4,   4,   4,   6,   2,   6,   6,   2,   5,   1,   6,   2,   4,   5,   2,   7,   4,   3,   3,   4,   5,   7,   3,   9,   5,   7,   2,   2,   3,   3,   3,   2,   1,   8,   1,   0,   1,   5,   5,   1,   3,   1,   8,   7,   2,   2,   2,   1,   2,   3,   3,   0,   3,   3,   3,   6,   6,   1,   3,   5,   5,   0,   6,   4,   1,   3,   3,   3,   3,   8,   3,   3,   1,   1,   4,   2,   3,   4,   1,   4,   4,   3,   4,   4,   8,   6,   1,   3,   3,   3,   1,   1,   1,   4,   0,   1,   0,   1,   4,   3,   1,   3,   2,   3,   1,   4,   2,   0,   3,   5,   4,   3,   1,   5,   4,   5,   3,   3,   5,   2,   1,   0,   1,   0,   1,   3,   2,   3,   2,   4,   2,   3,   3,   4,   6,   1,   5,   3,   2,   2,   2,   1,   0,   2,   0,   2,   2,   2,   3,   0,   0,   1,   3,   2,   1,   3,   3,   1,   0,   0,   0,   6,   3,   3,   4,   2,   3,   0,   1,   1,   2,   2,   2,   1,   3,   0,   1,   0,   0,   0,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0])
rate2 = np.array([393, 384, 372, 380, 392, 385, 273])
rate2 = np.array([391, 393, 384, 372, 380, 273])
#t_vz  = np.array([0  , 4  , 8  , 12 , 13 , 14 , 15 , 16 , 20 , 24 , 28 ])
t_vz  = np.array([-4 , 0  , 4  , 8  , 12  , 16])#letzter wert war glaube ich bei 20 und nicht 16

#andere daten mit den alten breiten, die lagen bei etwa 20ns oder so
t_vz1  = np.array([0  , 4  , 8  , 12 , 13 , 14 , 15 , 16 , 20 , 24 , 28 ])
rate  = np.array([151, 180, 175, 159, 179, 182, 189, 205, 188, 188, 158])



#def g(x, sigma_0, sigma_1):
#    return sigma_0 * np.exp(-sigma_1*(x-393)**2)
#
#params_eich, covariance_matrix_eich = curve_fit(g, t_vz, rate2)
#t_lin = np.linspace(-5,30,1000)

#plt.plot(t_lin, g(t_lin,*noms(params_eich)),  'b-', label='Fit' )
plt.plot(t_vz, rate2 , 'r.', label='Messdaten')
plt.xlabel(r'$t_\text{VZ} \,/\, \si{\nano\second}$')
plt.ylabel(r'Impulse')
plt.xlim(-4.5,30)
plt.ylim(0,405)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/verzoegerung.pdf')
plt.clf()

# Kanäle (eichung):
kanal_nummer = np.array([ 45, 90, 135, 179, 224, 269, 313, 358, 403  ]) #statet bei 0, d.h. der erste Kanal ist 0, der letzte 511
t_s = np.array([ 1,2,3,4,5,6,7,8,9 ]) * 10**(-6) # zum kanal n gehört jeweils diese Lebenszeit


### KALIBRIERUNG
# Lineare Regression: Eichgerade, um Kanälen die jeweilige Lebensdauer zuzuordnen

def f(x, a, b):
    return a * x + b

params_eich, covariance_matrix_eich = curve_fit(f, kanal_nummer, t_s)
errors_eich = np.sqrt(np.diag(covariance_matrix_eich))

def tau(kanal):
    ''' Gebe Kanalnummer ein - Erhalte dazugehörige Lebensdauer '''
    return f(kanal, *params_eich)

tau_x = tau(np.linspace(0,511,512)) # Zu den Kanälen 0,1,...,511 stehen in diesem Array nun die Lebensdauern
kanal_plot = np.linspace(0,511,512)


print("Abgeschnitten bei: ", tau_x[511-16])

T_such = ufloat( tau_x[511-16], tau_x[511]-tau_x[510])

write('build/T_such.tex', make_SI(T_such * 10**6, r'\micro\second', figures=1))



plt.xlabel('Kanal')
plt.ylabel(r'$\tau \,/\, \si{\micro\second}$')

plt.plot(kanal_plot, f(kanal_plot, *noms(params_eich))*10**6, 'b-', label='Fit')
plt.plot(kanal_nummer, t_s*10**6, 'r.', label='Messdaten')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/eichung.pdf')
plt.clf()

m_val = ufloat(params_eich[0], errors_eich[0])
b_val = ufloat(params_eich[1], errors_eich[1])

write('build/m_val.tex', make_SI(m_val * 10**6, r'\micro\second', figures=1))
write('build/b_val.tex', make_SI(b_val * 10**6, r'\micro\second', figures=1))

#### FITTE EXPONENTIALFUNKTION

def exp_dist(t, N_0, lambd, U):
    return N_0  * np.exp(-lambd * t) + U
    #fitten mit lambda extra funktioniert nicht. buh.

## Gebe Lebensdauern der ersten drei Kanäle aus

print("ersten drei Kanäle:", tau_x[0],tau_x[1],tau_x[2])


## Lösche schlechte Werte
tau_x_fit = tau_x[3:-16]
data_fit = data[3:-16]
tau_x_raus = np.append( tau_x[0:3], tau_x[-16:] )
data_fit_raus = np.append( data[0:3], data[-16:] )


params_fit_gew, covariance_matrix_fit_gew = curve_fit(exp_dist, tau_x_fit, data_fit, sigma = np.sqrt(data_fit)+1*(data_fit==0)) # Workaround für sigma: Bei allen Nullmessungen packe eine eins dazu, damit gewichteter Fit möglich ,  + 1*(data==0)
errors_fit_gew = np.sqrt(np.diag(covariance_matrix_fit_gew))

params_fit, covariance_matrix_fit = curve_fit(exp_dist, tau_x_fit, data_fit)
errors_fit = np.sqrt(np.diag(covariance_matrix_fit))



### Mister Monte-Carlo (presented by Super Mario Copy-Pasta)

#x_plot_up = np.zeros(len(tau_x))
#x_plot_down = np.zeros(len(tau_x))
#a_mc = random.multivariate_normal(params_fit_gew, covariance_matrix_fit_gew, 10000)
#
#
#for i, val in enumerate(tau_x):
#    mc_values = []
#
#    for k in a_mc:
#        mc_values.append(exp_dist(val, *k))
#
#    mc_mean = np.mean(mc_values)
#    mc_std = np.std(mc_values)
#
#    x_plot_up[i] = mc_mean + 2*mc_std
#    x_plot_down[i] = mc_mean - 2*mc_std

### Mister Not-Monte-Carlo (presented by Super Mario Copy-Pasta)


def exp_dist_unp(t, N_0, lambd, U):
    return N_0  * exp(-lambd * t) + U

x_plot_up = np.zeros(len(tau_x))
x_plot_down = np.zeros(len(tau_x))

N_0_with_err = ufloat(params_fit_gew[0], errors_fit_gew[0])
lambd_with_err = ufloat(params_fit_gew[1], errors_fit_gew[1])
U_with_err = ufloat(params_fit_gew[2], errors_fit_gew[2])

for i, val in enumerate(tau_x):

    tmp = exp_dist_unp(val, N_0_with_err, lambd_with_err, U_with_err)

    x_plot_up[i] = tmp.n + 2*tmp.s
    x_plot_down[i] = tmp.n - 2*tmp.s

### Plot gewichtet

plt.errorbar(tau_x_raus*10**6, data_fit_raus, fmt='r.', yerr=np.sqrt(data_fit_raus) , label='Messdaten nicht berücksichtigt',  linewidth=1,  markersize='1', capsize=1)
plt.errorbar(tau_x_fit*10**6, data_fit, fmt='k.', yerr=np.sqrt(data_fit) , label='Messdaten berücksichtigt',  linewidth=1,  markersize='1', capsize=1)

plt.ylabel(r'$N$')
plt.xlabel(r'$\tau \,/\, \si{\micro\second}$')

plt.plot(tau_x*10**6, exp_dist(tau_x, *noms(params_fit_gew)), 'b-', label='Fit')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/expfit_gew.pdf')

plt.clf()

## Plot ungewichtet

plt.errorbar(tau_x_raus*10**6, data_fit_raus, fmt='r.', yerr=np.sqrt(data_fit_raus) , label='Messdaten nicht berücksichtigt',  linewidth=1,  markersize='1', capsize=1)
plt.errorbar(tau_x_fit*10**6, data_fit, fmt='k.', yerr=np.sqrt(data_fit) , label='Messdaten berücksichtigt',  linewidth=1,  markersize='1', capsize=1)

plt.ylabel(r'$N$')
plt.xlabel(r'$\tau \,/\, \si{\micro\second}$')

plt.plot(tau_x*10**6, exp_dist(tau_x, *noms(params_fit)), 'b-', label='Fit')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/expfit.pdf')

plt.clf()




## Plot gewichtet (Robert Edition)

#plt.errorbar(tau_x_raus[data_fit_raus>1]*10**6, data_fit_raus[data_fit_raus>1], fmt='r.', yerr=np.sqrt(data_fit_raus[data_fit_raus>1]),  linewidth=1,  markersize='1', capsize=1 , label='Messdaten nicht berücksichtigt')
plt.errorbar(tau_x_fit[data_fit>1]*10**6, data_fit[data_fit>1], fmt='k.', yerr=np.sqrt(data_fit[data_fit>1]) ,  linewidth=1,  markersize='1', capsize=1, label='Messdaten berücksichtigt')

plt.plot(tau_x*10**6, exp_dist(tau_x, *noms(params_fit_gew)), 'b-', label='Fit')

plt.ylabel(r'$N$')
plt.xlabel(r'$\tau \,/\, \si{\micro\second}$')

plt.yscale('log')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/expfit_gew_robert.pdf')

plt.clf()

## Plot ungewichtet (Robert Edition)

#plt.errorbar(tau_x_raus[data_fit_raus>1]*10**6, data_fit_raus[data_fit_raus>1], fmt='r.', yerr=np.sqrt(data_fit_raus[data_fit_raus>1]) , label='Messdaten nicht berücksichtigt',  linewidth=1,  markersize='1', capsize=1)
plt.errorbar(tau_x_fit[data_fit>1]*10**6, data_fit[data_fit>1], fmt='k.', yerr=np.sqrt(data_fit[data_fit>1]) , label='Messdaten berücksichtigt',  linewidth=1,  markersize='1', capsize=1)


plt.plot(tau_x*10**6, exp_dist(tau_x, *noms(params_fit)), 'b-', label='Fit')

plt.xlabel(r'$\tau \,/\, \si{\micro\second}$')
plt.legend(loc='best')
plt.yscale('log')
plt.ylabel(r'$N$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/expfit_robert.pdf')

plt.clf()



### werte ungewichtet
N_0_val = ufloat(params_fit[0], errors_fit[0])
lambd_val = ufloat(params_fit[1], errors_fit[1])
U_val = ufloat(params_fit[2], errors_fit[2])

write('build/N_0_val.tex', make_SI(N_0_val, r'', figures=2))
write('build/lambd_val.tex', make_SI(lambd_val / (10**6), r'\per\micro\second', figures=2))
write('build/U_val.tex', make_SI(U_val, r'', figures=2))

tau = 1/lambd_val

write('build/tau.tex', make_SI(tau*10**6, r'\micro\second', figures=1))

### werte gewichtet

N_0_val_gew = ufloat(params_fit_gew[0], errors_fit_gew[0])
lambd_val_gew = ufloat(params_fit_gew[1], errors_fit_gew[1])
U_val_gew = ufloat(params_fit_gew[2], errors_fit_gew[2])

write('build/N_0_val_gew.tex', make_SI(N_0_val_gew, r'', figures=2))
write('build/lambd_val_gew.tex', make_SI(lambd_val_gew / (10**6), r'\per\micro\second', figures=2))
write('build/U_val_gew.tex', make_SI(U_val_gew, r'', figures=2))

tau_gew = 1/lambd_val_gew

write('build/tau_gew.tex', make_SI(tau_gew*10**6, r'\micro\second', figures=1))


tau_lit = ufloat(2.196*10**(-6), 0.000002*10**(-6))
write('build/tau_lit.tex', make_SI(tau_lit*10**6, r'\micro\second', figures=1))

abw = np.abs(tau_lit - tau_gew)/tau_lit * 100

write('build/tau_lit_abw.tex', make_SI(abw.n, r'\percent', figures=1))

print(N_0_val)
print(lambd_val)
print(U_val)
print(tau)


### Nur Sigmaintervall und Punkte

plt.plot(tau_x_raus*10**6, data_fit_raus, 'r.', label='Messdaten nicht berücksichtigt', markersize=1)
plt.plot(tau_x_fit*10**6, data_fit, 'k.', label='Messdaten berücksichtigt', markersize=1)
plt.fill_between(tau_x*10**6, x_plot_up,  x_plot_down, interpolate=True, alpha=0.7, color='b',linewidth=0.0, zorder=50, label = r'$2\sigma-\text{Intervall}$') #Fehlerdings
plt.ylabel(r'$N$')
plt.xlabel(r'$\tau \,/\, \si{\micro\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/expfit_sigma.pdf')

### Bestimmung Untergrund (ist das richtig?!!?)

N_ges = ufloat(2882767, np.sqrt(2882767))
write('build/N_ges.tex', make_SI(N_ges, r'', figures=1))
t_ges = 160147
#T_such = 15*10**(-6) #suchzeit siehe oben

mu_mittel = N_ges/t_ges * T_such # so viele Myonen durchqueren den Tank im Mittel in der Suchzeit
# mu_mittel ist dann auch der Erwartungswert der Poissionverteilung

mu_folgend = mu_mittel * exp(-mu_mittel)
#poissionverteilung mit lambda = mu_mittel und n=1

mu_gesamt_fehl = mu_folgend * N_ges

U_theo = mu_gesamt_fehl/(512-3-17) # aufgeteilt auf 512 Kanäle

write('build/U_theo.tex', make_SI(U_theo, r'', figures=1))
