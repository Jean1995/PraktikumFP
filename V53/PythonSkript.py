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

def f3(x, a, b, c):
    return a*(b-x)**2 + c

V_1_fake = np.array([205, 207.5, 215, 225, 230]) # in Volt
A_1_fake = np.array([0, 9,  18, 9, 0])
V_1 = np.array([205, 215, 230]) # in Volt
A_1 = np.array([0, 18, 0])
V_2_fake = np.array([120, 124, 140, 147.5, 150])
A_2_fake = np.array([0, 5, 20,10,  0])
V_2 = np.array([120, 140, 150])
A_2 = np.array([0, 20, 0])
V_3_fake = np.array([70, 75, 85, 89,  90])
A_3_fake = np.array([0, 4, 14, 6, 0])
V_3 = np.array([70, 85, 90])
A_3 = np.array([0, 14, 0])

mode = np.array([1,2,3])
V_null = np.array([215, 140, 85])
V_eins = np.array([230, 150, 90])
V_zwei = np.array([205, 120, 70])
A = np.array([18, 20, 14])
f = np.array([9000, 9005, 9010]) # megahertz

write('build/md.tex', make_table([mode, V_null, V_eins, V_zwei, A, f],[0, 0, 0, 0, 0, 0]))
write('build/moden.tex', make_full_table(
     'Messdaten Modenbestimmung.',
     'tab:moden',
     'build/md.tex',
     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                               # die Multicolumns sein sollen
     [r'Mode',
     r'$V_0 \:/\: \si{\volt}$',
     r'$V_1 \:/\: \si{\volt}$',
     r'$V_2 \:/\: \si{\volt}$',
     r'$A \:/\: \si{\milli\volt}$',
     r'$f \:/\: \si{\mega\hertz}$']))


plt.plot(V_1, A_1, 'rx', label='1. Modus')
plt.plot(V_2, A_2, 'bx', label='2. Modus')
plt.plot(V_3, A_3, 'yx', label='3. Modus')

plt.rcParams['lines.linewidth'] = 1


x_plot1 = np.linspace(np.min(V_1), np.max(V_1))
yinterp1 = spline(V_1_fake, A_1_fake, x_plot1)
plt.plot(x_plot1, yinterp1, 'r--')

x_plot2 = np.linspace(np.min(V_2), np.max(V_2))
yinterp2 = spline(V_2_fake, A_2_fake, x_plot2)
plt.plot(x_plot2, yinterp2, 'b--')

x_plot3 = np.linspace(np.min(V_3), np.max(V_3))
yinterp3 = spline(V_3_fake, A_3_fake, x_plot3)
plt.plot(x_plot3, yinterp3, 'y--')


plt.xlabel(r'$V_\text{Ref}$')
plt.ylabel(r'Leistung')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/moden.pdf')
plt.clf()

## Versuch 1 - Elektronische Abstimmung

f_el = np.array([9000, 8985, 9017])*10**6
f_el_err = np.array([1, 1, 1])*10**6
V_el = np.array([215, 205, 225])
V_el_err = np.array([3,3,3])

write('build/brt.tex', make_table([V_el, f_el*10**(-6)],[0, 0]))
write('build/breite.tex', make_full_table(
     'Messdaten Elektronische Abstimmung.',
     'tab:breite',
     'build/brt.tex',
     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                               # die Multicolumns sein sollen
     [r'$V_\text{Ref} \:/\: \si{\volt}$',
     r'$f \:/\: \si{\mega\hertz}$']))

f_el = unp.uarray(f_el, f_el_err)
V_el = unp.uarray(V_el, V_el_err)

bandbreite = np.max(f_el) - np.min(f_el)
write('build/B.tex', make_SI(bandbreite*10**(-6),  r'\mega\hertz', figures=1))

abstimmempf = (np.max(f_el) - np.min(f_el)) / (np.max(V_el) - np.min(V_el) )
write('build/A.tex', make_SI(abstimmempf*10**(-6),  r'\mega\hertz\per\volt', figures=1))

print("Bandbreite =", bandbreite)
print("Abstimmempfindlichkeit =", abstimmempf)



## Versuchs 2 - Wellenlängen

a = ufloat( 22.860, 0.046 ) *10**(-3)
min1 = ufloat(115.9, 0.1) * 10**(-3)
min2 = ufloat(90.8, 0.1) * 10**(-3)
delta_min = min1-min2

lambda_g = 2*delta_min
print(type(lambda_g))
write('build/lambda_g.tex', make_SI(lambda_g*10**(3),  r'\milli\metre', figures=1))

f = const.c * unp.sqrt( (1/lambda_g)**2 + (1/(2*a))**2 )
write('build/f.tex', make_SI(f*10**(-6),  r'\mega\hertz', figures=1))

### Versuch 2 Dämpfung

SWR = np.array([0,2,4,6,8,10])
dmpf_mm = np.array([0.6, 1.32, 1.61, 1.95, 2.2, 2.59]) # milli meter
dmpf_mm_err = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) # milli meter
SWR_real = np.array([ 1, 3.5, 5, 7.5, 9.5, 12.5  ])

write('build/dmp.tex', make_table([SWR, dmpf_mm],[0, 2]))
write('build/daempfung.tex', make_full_table(
     'Messdaten Dämpfung.',
     'tab:daempfung',
     'build/dmp.tex',
     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                               # die Multicolumns sein sollen
     [r'$\text{SWR-Ausschlag} \:/\: \si{\decibel}$',
     r'$\text{Schraubenstellung} \:/\: \si{\milli\metre}$']))


plt.plot(dmpf_mm, SWR, 'rx', label='Messdaten')
plt.plot(dmpf_mm, SWR_real, 'gx', label='Theoriewerte')

#plt.errorbar(dmpf_mm, SWR, fmt='rx', xerr=dmpf_mm_err, label='Messdaten')

def fd(x, a, b, c):
    return a*np.exp(x*b) + c

params_d, cov_d = curve_fit(fd, dmpf_mm, SWR)
params_d_real, cov_d_real = curve_fit(fd, dmpf_mm, SWR_real)


x_plotd = np.linspace(np.min(dmpf_mm), np.max(dmpf_mm))


plt.plot(x_plotd, fd(x_plotd, *noms(params_d)), 'r--')
plt.plot(x_plotd, fd(x_plotd, *noms(params_d_real)), 'g--')


plt.xlabel(r'$\text{Schraubenstellung} \:/\: \si{\milli\metre}$')
plt.ylabel(r'$\text{SWR-Ausschlag} \:/\: \si{\decibel}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/daempfung.pdf')
plt.clf()

dmpf_mm = unp.uarray(dmpf_mm, dmpf_mm_err)

### Versuch 3 direkte Methode

sondentiefe = np.array([3, 5, 7, 9]) * 10**(-3) #in meter
SWS = np.array([1.05, 1.25, 1.85, 4.0])

write('build/wgk.tex', make_table([sondentiefe*10**(3), SWS],[0, 2]))
write('build/welligkeit.tex', make_full_table(
     'Messung der Welligkeit über die direkte Methode.',
     'tab:welligkeit',
     'build/wgk.tex',
     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                               # die Multicolumns sein sollen
     [r'$\text{Sondentiefe} \:/\: \si{\milli\metre}$',
     r'$S$']))

### Versuch 3 3db Methode

d1 = ufloat(112.7, 0.1) * 10**(-3)
d2 = ufloat(110.9, 0.1) * 10**(-3)

uff = unp.sqrt( 1 + 1/(unp.sin( np.pi*(d1-d2)/lambda_g )**2 ) )
uff = ufloat(unp.nominal_values(uff), unp.std_devs(uff))



write('build/S_3db.tex', make_SI(uff, r'\decibel', figures=2))

### Versuch 3 Abschwächer Methode

A2 = ufloat(38,1)
A1 = ufloat(20,1)
S_abs = (A2-A1)/2

write('build/S_abs.tex', make_SI(S_abs, r'\decibel', figures=2))

# Diskussion

### Versuch 3

abw_S = 100*(S_abs.n - uff.n)/uff.n
write('build/abw_S.tex', make_SI(abw_S, r'\percent', figures=1))

### Versuch 2

abw_f = np.abs((f.n-9000*10**6)/(9000*10**6)*100)
write('build/abw_f.tex', make_SI(abw_f, r'\percent', figures=1))
