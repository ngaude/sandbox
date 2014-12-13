# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 00:57:19 2014

@author: Utilisateur
"""

import matplotlib.pyplot as plt
import math

class Simu:
    # revalorisation mensuelle des taux d'interets
    i_plac = [0.0175]*1000 # croissance annuelle placement financier
    i_immo = [-0.0050]*1000 # croissance annuelle immobilier
    i_loc = [0.005]*1000 # croissance annuelle du loyer
    
    # charge proprietaire vs locataire
    c_immo_p = 500
    c_immo_l = 220

    # caracteristique d'un pret immo à 2,7%/an duree 12 ans    
    i_pret = 0.027
    d_pret = 144

    # loyer appartement
    i_loyer = 1910
    
    def __init__(self):
        self.cap_plac = 570000
        self.cap_immo = 0
        self.salaire = 3600
        self.mois = 0
        self.locataire = True
        self.loyer = Simu.i_loyer
        self.mensualite = 0
        self.crd = 0
        self.charges = Simu.c_immo_l
        
    def iteration_mensuelle(self):
        self.mois += 1
        # on touche le salaire
        self.cap_plac += self.salaire
        # on paye le loyer
        self.iter_locataire()
        # on rembourse l'appart
        self.iter_proprio()
        # on paye les charges
        self.iter_charges()
        # on touche les interets du capital
        self.iter_capital()
        return self.cap_plac+self.cap_immo-self.crd
                        
    def iter_locataire(self):
        if (self.locataire == False):
            return
        # augmentation mensuelle du loyer
        self.loyer = self.loyer * (1+Simu.i_loc[self.mois]/12.0)
        self.cap_plac -= self.loyer
        return
    def iter_proprio(self):
        if (self.locataire == True):
            return
        # on arrete de payer le pret quand le crd passe negatif
        if (self.crd < 0):
            return
        # on ajoute les interets au crd
        interet  = self.crd * Simu.i_pret/12.0
        self.crd += interet
        # on retire la mensualité du crd       
        self.crd -= self.mensualite  
        # on prend la mensualité depuis le capital
        self.cap_plac -= self.mensualite
        # on modifie eventuellement son capital immobilier
        self.cap_immo *= (1+Simu.i_immo[self.mois]/12.0)
        return
        
    def iter_charges(self):
        self.cap_plac -= self.charges
        return
    def iter_capital(self):
        self.cap_plac *= (1+Simu.i_plac[self.mois]/12.0)
        return
    def event_achat(self,prix_appart,prix_travaux):
        self.locataire = False
        self.loyer = 0
        self.charges = Simu.c_immo_p
        # payer le prix de l'appart plus le notaire 7%
        self.cap_plac -= prix_appart * 1.07
        self.cap_plac -= prix_travaux
        # le capital immobilier vaut le prix de l'appartement
        self.cap_immo = prix_appart
        if (self.cap_plac < 30000):
            self.crd = 30000 - self.cap_plac
            # calcul du cout total du pret sur la duree
            i_m = Simu.i_pret/12.0
            self.mensualite = (self.crd*i_m)/(1-math.pow(1+i_m,-Simu.d_pret))
        # on rajoute le montant du pret au capital placé qui sert a l'achat
        self.cap_plac += self.crd
        return
a = Simu()
capa = [a.iteration_mensuelle() for m in range(0,144)]
b = Simu()
b.event_achat(760000,50000)
capb = [b.iteration_mensuelle() for m in range(0,144)]

plt.plot(capa,label = 'locataire')
plt.plot(capb,label = 'proprietaire')
plt.legend(loc = 'upper left')      
        
        