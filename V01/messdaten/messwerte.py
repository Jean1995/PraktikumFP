import numpy as np

t_vz  = np.array([0  , 4  , 8  , 12 , 13 , 14 , 15 , 16 , 20 , 24 , 28 ])
rate  = np.array([151, 180, 175, 159, 179, 182, 189, 205, 188, 188, 158])
#20sec
rate2 = np.array([393, 384, 372, 380, 392, 385, 273, ])

T_vz = 10 #ns rechte Seite

# ca 17% werden durch die Koinzidenz rausgefiltert
# T_S = 15µs


### Time analyser: 10 µs
