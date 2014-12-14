# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 00:57:19 2014

@author: Utilisateur
"""

import matplotlib.pyplot as plt
import numpy as np
import math

duree_etude = 300

def taux_variable(tx_debut,tx_fin,mois_debut, mois_fin):
    '''
    genere un taux de credit variable selon :
    taux constant a tx_debut du mois 0 à mois mois_debut
    taux variable linerairement de tx_debut à tx_fin entre mois_debut et mois_fin
    taux constant a tx_fin du mois mois_fin au dela
    '''    
    tx = [tx_debut]*mois_debut 
    tx += list(np.linspace(tx_debut,tx_fin,mois_fin-mois_debut))
    tx += [tx_fin]*(duree_etude+1-mois_fin)
    return tx


class Simu:
    # revalorisation mensuelle des taux d'interets

#     scenario sortie de crise dans 3 ans, sur 7ans
#    i_plac = taux_variable(1.75,4.5,36,120)
#    i_immo = taux_variable(-0.5,3,54,138)
#    i_loc = taux_variable(0.5,2,46,130)
#    i_pret = taux_variable(2.7,5.1,36,120)

    i_plac = [1.75]*(duree_etude+1) # croissance annuelle placement financier
    i_immo = [-0.5]*(duree_etude+1) # croissance annuelle immobilier
    i_loc = [0.5]*(duree_etude+1) # croissance annuelle du loyer
    i_pret = [2.7]*(duree_etude+1) # croissance du taux d'emprunt
    
    # charge proprietaire vs locataire
    c_immo_p = 500
    c_immo_l = 220

    # caracteristique d'un pret immo à 2,7%/an duree 12 ans    
    d_pret = 144

    # loyer appartement
    i_loyer = 1910
    
    def __init__(self):
        self.cap_plac = 570000
        self.cap_immo = 0
        self.salaire = 4500
        self.mois = 0
        self.locataire = True
        self.loyer = Simu.i_loyer
        self.mensualite = 0
        self.crd = 0
        self.taux_pret = 0
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
        self.loyer = self.loyer * (1+Simu.i_loc[self.mois]/1200.0)
        self.cap_plac -= self.loyer
        return
    def iter_proprio(self):
        if (self.locataire == True):
            return
        # on arrete de payer le pret quand le crd passe negatif
        if (self.crd > 0):
            # on ajoute les interets au crd
            interet  = self.crd * self.taux_pret/1200.0
            self.crd += interet
            # on retire la mensualité du crd       
            self.crd -= self.mensualite  
            # on prend la mensualité depuis le capital
            self.cap_plac -= self.mensualite
        # on renumere son capital immobilier
        self.cap_immo *= (1+Simu.i_immo[self.mois]/1200.0)
        return
        
    def iter_charges(self):
        self.cap_plac -= self.charges
        return
    def iter_capital(self):
        self.cap_plac *= (1+Simu.i_plac[self.mois]/1200.0)
        return
    def event_achat(self,prix_appart,prix_travaux):
        self.locataire = False
        self.loyer = 0
        self.charges = Simu.c_immo_p
        # on valide au moment de l'achat, le taux courant
        self.taux_pret = Simu.i_pret[self.mois]
        # payer le prix de l'appart plus le notaire 7%
        self.cap_plac -= prix_appart * 1.07
        self.cap_plac -= prix_travaux
        # le capital immobilier vaut le prix de l'appartement
        self.cap_immo = prix_appart
        if (self.cap_plac < 30000):
            self.crd = 30000 - self.cap_plac
            # calcul du cout total du pret sur la duree
            i_m = self.taux_pret/1200.0
            self.mensualite = (self.crd*i_m)/(1-math.pow(1+i_m,-Simu.d_pret))
        # on rajoute le montant du pret au capital placé qui sert a l'achat
        self.cap_plac += self.crd
        return

        
        
a = Simu()
capa = [a.iteration_mensuelle() for m in range(duree_etude)]
b = Simu()
b.event_achat(760000,30000)
capb = [b.iteration_mensuelle() for m in range(duree_etude)]

# attente de 3 ans avant achat du bien
c = Simu()
attente_achat = 50
capc = [c.iteration_mensuelle() for m in range(attente_achat)]
c.event_achat(760000,30000)
capc += [c.iteration_mensuelle() for m in range(duree_etude - attente_achat)]

def diff_list(a,b):
    return [ea - eb for (ea,eb) in zip(a,b)]

plt.figure(1)
plt.subplot(1,2,1)
plt.plot(capa,label = 'locataire')
plt.plot(capb,label = 'proprietaire')
plt.plot(capc,label = 'proprio '+ str(attente_achat/12) +' ans')
#plt.plot(diff_list(capa,capa),label = 'locataire')
#plt.plot(diff_list(capb,capa),label = 'proprietaire')
#plt.plot(diff_list(capc,capa),label = 'proprio '+ str(attente_achat/12) +' ans')
plt.legend(loc = 'upper left',prop={'size':8})
plt.subplot(1,2,2)
plt.plot(Simu.i_plac,label = 'taux capital')
plt.plot(Simu.i_immo,label = 'prix immo')
plt.plot(Simu.i_loc,label = 'taux loyer')
plt.plot(Simu.i_pret,label = 'taux emprunt')
plt.legend(loc = 'lower right',prop={'size':8})
plt.show()

        