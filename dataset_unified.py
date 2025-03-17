
import torch
import numpy as np

class h5pyDataset(torch.utils.data.Dataset):
    def __init__(self, labels_pvi, labels_pvi_biatrial, labels_pvira, labels_pvila, psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, \
                 mask_la_pvi, mask_ra_pvi, mask_la_pvi_biatrial, mask_ra_pvi_biatrial): #, mask_la_pvira, mask_ra_pvira,mask_la_pvila, mask_ra_pvila):
        self.psd_la = psd_la
        self.psd_ra = psd_ra
        self.df_la = df_la
        self.df_ra = df_ra
        self.fibre_la = fibre_la
        self.fibre_ra = fibre_ra
        self.mask_la_pvi = mask_la_pvi
        self.mask_ra_pvi = mask_ra_pvi
        self.labels_pvi = labels_pvi
        self.labels_pvira = labels_pvira
        self.labels_pvila = labels_pvila
        self.mask_la_pvi_biatrial = mask_la_pvi_biatrial
        self.mask_ra_pvi_biatrial = mask_ra_pvi_biatrial
        self.labels_pvi_biatrial = labels_pvi_biatrial        

    def __len__(self):
        return len(self.psd_la) 

    def __getitem__(self, index):

        mask_la_pvi = np.array(self.mask_la_pvi[index][None, :])
        mask_ra_pvi = np.array(self.mask_ra_pvi[index][None, :])
        mask_la_pvi_biatrial = np.array(self.mask_la_pvi_biatrial[index][None, :])
        mask_ra_pvi_biatrial = np.array(self.mask_ra_pvi_biatrial[index][None, :])
        labels=[]
        labels.append(self.labels_pvi[index])
        labels.append(self.labels_pvira[index])
        labels.append(self.labels_pvila[index])
        labels.append(self.labels_pvi_biatrial[index])
        psd_la = np.array(self.psd_la[index][None, :])
        psd_ra = np.array(self.psd_ra[index][None, :])
        df_la = np.array(self.df_la[index][None, :])
        df_ra = np.array(self.df_ra[index][None, :])
        fibre_la = np.array(self.fibre_la[index][None, :])
        fibre_ra = np.array(self.fibre_ra[index][None, :])
            
        return {'psd_la': psd_la,
                'psd_ra': psd_ra,
                'df_la': df_la,
                'df_ra': df_ra,
                'fibre_la': fibre_la,
                'fibre_ra': fibre_ra,
                'mask_la_pvi': mask_la_pvi,
                'mask_ra_pvi': mask_ra_pvi,
                'mask_la_pvi_biatrial': mask_la_pvi_biatrial,
                'mask_ra_pvi_biatrial': mask_ra_pvi_biatrial,
                'labels': np.array(labels)}