import numpy as np
import torch

# Cose da fare:
# 0. Capire bene che modello utilizzare
# 1. sulla base di questo capire quali funzioni di preprocessing implementare (crop, resize, normalizzazione, ecc.) :
# 2. Fare classe Rastro_Dataset che preveda un argomento agent_idx (dove agent_idx è l'indice dell'agente in [0,1,2,3,4,5]).
#    2.1 Se agent_idx = -1 allora il dataset deve essere CENTRALIZZATO (contenere tutti i dati)
#    2.2 Se agent_idx = i \in [0,1,2,3,4,5] allora il dataset deve contenere solo i dati dell'agente i-esimo
# 3. Implementare primi esperimenti con dataset centralizzato
# To be continued...


class Rastro_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pass
