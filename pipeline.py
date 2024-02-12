

import pytorch_lightning
from monai.utils import set_determinism
from monai.networks.nets import DenseNet121, EfficientNetBN
from monai.data import list_data_collate, DataLoader
import torch
# import pandas as pd
import numpy as np
import os
import h5py
import torch.nn as nn
from monai.metrics import compute_roc_auc
from torchmetrics.classification import BinaryAccuracy
import torchmetrics
import numpy as np
import torch
from typing import Any
import torch.nn.functional as F

from models import *
from dataset import h5pyDataset

class Net(pytorch_lightning.LightningModule):
    def __init__(self, result_dir, **config: Any):
        super().__init__()
        if config['training']['model'] == 'DenseNet121':
            self.model_1 = DenseNet121(spatial_dims=2, in_channels=4, out_channels=config['training']['out_channels'])
            self.model_2 = DenseNet121(spatial_dims=2, in_channels=4, out_channels=config['training']['out_channels'])
        if config['training']['model'] == 'EfficientNetBN':
            self.model_1 = EfficientNetBN("efficientnet-b0", pretrained=False, progress=False, spatial_dims=2, in_channels=4, num_classes=config['training']['out_channels'])
            self.model_2 = EfficientNetBN("efficientnet-b0", pretrained=False, progress=False, spatial_dims=2, in_channels=4, num_classes=config['training']['out_channels'])
        self.fc = nn.Linear((config['training']['out_channels']+1)*(config['training']['out_channels']+1), config['training']['fc_channels'])
        self.dropout = nn.Dropout(p=0.1)
        self.layer_out = nn.Linear(config['training']['fc_channels'], 1)        
        self.conv_stack= conv_(4,1)    
        self.ln = nn.Linear(config['training']['out_channels']*2, 1)

        self._config = config
        self.result_dir = result_dir
        self.relu = nn.ReLU()     

        if config['training']['loss'] == 'BCE':       
            self.loss_function = torch.nn.BCELoss()
        if config['training']['loss'] == 'BCELogits':       
            self.loss_function = torch.nn.BCEWithLogitsLoss()    

        self.max_epochs = config['training']['max_epochs']
        self.check_val = config['training']['check_val']
        self.f1_metric = torchmetrics.F1Score(task="binary")
        self.accuracy = BinaryAccuracy()
        self.best_val_roc = 0
        self.best_val_f1 = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.f1_metrics = []
        self.rocauc = []
        self.acc = []

    def forward(self, psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, mask_la, mask_ra):
        
        la = torch.cat((psd_la, df_la, fibre_la, mask_la), dim=1)
        ra = torch.cat((psd_ra, df_ra, fibre_ra, mask_ra), dim=1)

        if self._config['training']['merging'] == 'MOAB':   
            x1 = self.model_1(la)
            x3 = self.model_2(ra)    

            # This is done to flatten the feature map from the MLP layer.
            x3 = x3.view(x3.size(0), -1)
                
            ## outer addition branch (appending 0)
            x_add = append_0(x1,x3,self._config)
            x_add = torch.unsqueeze(x_add, 1)
            ## outer subtraction branch (appending 0)
            x_sub = append_0_s(x1,x3,self._config)
            x_sub = torch.unsqueeze(x_sub, 1)

            ## outer product branch (appending 1)
            x_pro =append_1(x1,x3,self._config)
            x_pro = torch.unsqueeze(x_pro, 1)
            
            ## outer divison branch (appending 1)
            x_div =append_1_d(x1,x3,self._config)
            x_div = torch.unsqueeze(x_div, 1)
            
            ## combine 4 branches on the channel dim
            x = torch.cat((x_add,x_sub,x_pro,x_div),dim=1)
            #print('shape afr cat', x.shape)
            
            ## use a conv (1x1) 
            x = self.conv_stack(x)
            #print('shape after conv', x.shape)
            x = x.flatten(start_dim=1)
            
            #print('shape aftr flatten', x.shape)
            
            x = self.fc(x)
            #print('fc after combined', x.shape)
            x = self.dropout(x)
            x = self.layer_out(x)

        if self._config['training']['merging'] == 'concat':
            out_1 = self.model_1(la)
            out_2 = self.model_2(ra)   
            x = torch.cat((out_1, out_2), dim=1)
            x = self.relu(x)
            x = torch.sigmoid(self.ln(x))
            
        return x

    def prepare_data(self):
        # set up the correct data path
            
        cases = h5py.File(self._config['data']['data'] ,'r')
        
        match_psd_la = [s for s in list(cases.keys()) if 'psd_la' in s]
        psd_la = np.zeros(((len(match_psd_la),cases[match_psd_la[0]].shape[0],cases[match_psd_la[0]].shape[1])))
        for i,_ in enumerate(match_psd_la):
            psd_la[i,:,:] = cases[match_psd_la[i]]
          
        match_psd_ra = [s for s in list(cases.keys()) if 'psd_ra' in s]
        psd_ra = np.zeros(((len(match_psd_ra),cases[match_psd_ra[0]].shape[0],cases[match_psd_ra[0]].shape[1])))
        for i,_ in enumerate(match_psd_ra):
            psd_ra[i,:,:] = cases[match_psd_ra[i]]  
            
        match_df_la = [s for s in list(cases.keys()) if 'df_la' in s]
        df_la = np.zeros(((len(match_df_la),cases[match_df_la[0]].shape[0],cases[match_df_la[0]].shape[1])))
        for i,_ in enumerate(match_df_la):
            df_la[i,:,:] = cases[match_df_la[i]]
          
        match_df_ra = [s for s in list(cases.keys()) if 'df_ra' in s]
        df_ra = np.zeros(((len(match_df_ra),cases[match_df_ra[0]].shape[0],cases[match_df_ra[0]].shape[1])))
        for i,_ in enumerate(match_df_ra):
            df_ra[i,:,:] = cases[match_df_ra[i]]
            
        match_fibre_la = [s for s in list(cases.keys()) if 'fibre_la' in s]
        fibre_la = np.zeros(((len(match_fibre_la),cases[match_fibre_la[0]].shape[0],cases[match_fibre_la[0]].shape[1])))
        for i,_ in enumerate(match_fibre_la):
            fibre_la[i,:,:] = cases[match_fibre_la[i]]
          
        match_fibre_ra = [s for s in list(cases.keys()) if 'fibre_ra' in s]
        fibre_ra = np.zeros(((len(match_fibre_ra),cases[match_fibre_ra[0]].shape[0],cases[match_fibre_ra[0]].shape[1])))
        for i,_ in enumerate(match_fibre_ra):
            fibre_ra[i,:,:] = cases[match_fibre_ra[i]]
            
        #masks

        match_mask_la_pvi = [s for s in list(cases.keys()) if 'mask_la' in s if 'biatrial' not in s if 'pvila' not in s if 'pvira' not in s]
        mask_la_pvi = np.zeros(((len(match_mask_la_pvi),cases[match_mask_la_pvi[0]].shape[0],cases[match_mask_la_pvi[0]].shape[1])))
        for i,_ in enumerate(match_mask_la_pvi):
            mask_la_pvi[i,:,:] = cases[match_mask_la_pvi[i]]
          
        match_mask_ra_pvi = [s for s in list(cases.keys()) if 'mask_ra' in s if 'biatrial' not in s if 'pvila' not in s if 'pvira' not in s]
        mask_ra_pvi = np.zeros(((len(match_mask_ra_pvi),cases[match_mask_ra_pvi[0]].shape[0],cases[match_mask_ra_pvi[0]].shape[1])))
        for i,_ in enumerate(match_mask_ra_pvi):
            mask_ra_pvi[i,:,:] = cases[match_mask_ra_pvi[i]]        



        match_mask_la_pvi_biatrial = [s for s in list(cases.keys()) if 'mask_la' in s if 'biatrial' in s if 'pvila' not in s if 'pvira' not in s]
        mask_la_pvi_biatrial = np.zeros(((len(match_mask_la_pvi_biatrial),cases[match_mask_la_pvi_biatrial[0]].shape[0],cases[match_mask_la_pvi_biatrial[0]].shape[1])))
        for i,_ in enumerate(match_mask_la_pvi_biatrial):
            mask_la_pvi_biatrial[i,:,:] = cases[match_mask_la_pvi_biatrial[i]]
          
        match_mask_ra_pvi_biatrial = [s for s in list(cases.keys()) if 'mask_ra' in s if 'biatrial' in s if 'pvila' not in s if 'pvira' not in s]
        mask_ra_pvi_biatrial = np.zeros(((len(match_mask_ra_pvi_biatrial),cases[match_mask_ra_pvi_biatrial[0]].shape[0],cases[match_mask_ra_pvi_biatrial[0]].shape[1])))
        for i,_ in enumerate(match_mask_ra_pvi_biatrial):
            mask_ra_pvi_biatrial[i,:,:] = cases[match_mask_ra_pvi_biatrial[i]]    



        match_mask_la_pvira = [s for s in list(cases.keys()) if 'mask_la' in s if 'pvira' in s if 'pvila' not in s if 'biatrial' not in s]
        mask_la_pvira = np.zeros(((len(match_mask_la_pvira),cases[match_mask_la_pvira[0]].shape[0],cases[match_mask_la_pvira[0]].shape[1])))
        for i,_ in enumerate(match_mask_la_pvira):
            mask_la_pvira[i,:,:] = cases[match_mask_la_pvira[i]]
          
        match_mask_ra_pvira = [s for s in list(cases.keys()) if 'mask_ra' in s if 'pvira' in s if 'pvila' not in s if 'biatrial' not in s]
        mask_ra_pvira = np.zeros(((len(match_mask_ra_pvira),cases[match_mask_ra_pvira[0]].shape[0],cases[match_mask_ra_pvira[0]].shape[1])))
        for i,_ in enumerate(match_mask_ra_pvira):
            mask_ra_pvira[i,:,:] = cases[match_mask_ra_pvira[i]] 



        match_mask_la_pvila = [s for s in list(cases.keys()) if 'mask_la' in s if 'pvila' in s if 'biatrial' not in s if 'pvira' not in s]
        mask_la_pvila = np.zeros(((len(match_mask_la_pvila),cases[match_mask_la_pvila[0]].shape[0],cases[match_mask_la_pvila[0]].shape[1])))
        for i,_ in enumerate(match_mask_la_pvila):
            mask_la_pvila[i,:,:] = cases[match_mask_la_pvila[i]]
          
        match_mask_ra_pvila = [s for s in list(cases.keys()) if 'mask_ra' in s if 'pvila' in s if 'biatrial' not in s if 'pvira' not in s]
        mask_ra_pvila = np.zeros(((len(match_mask_ra_pvila),cases[match_mask_ra_pvila[0]].shape[0],cases[match_mask_ra_pvila[0]].shape[1])))
        for i,_ in enumerate(match_mask_ra_pvila):
            mask_ra_pvila[i,:,:] = cases[match_mask_ra_pvila[i]]    

        #labels                  
                      
        match_labels_pvi = [s for s in list(cases.keys()) if 'label' in s if 'biatrial' not in s if 'RA' not in s if 'LA' not in s]
        labels_pvi = np.zeros(len(match_labels_pvi))
        for i,_ in enumerate(match_labels_pvi):
            labels_pvi[i] = cases[match_labels_pvi[i]][0]        

        match_labels_pvi_biatrial = [s for s in list(cases.keys()) if 'label' in s if 'biatrial' in s]
        labels_pvi_biatrial = np.zeros(len(match_labels_pvi_biatrial))
        for i,_ in enumerate(match_labels_pvi_biatrial):
            labels_pvi_biatrial[i] = cases[match_labels_pvi_biatrial[i]][0] 

        match_labels_pvira = [s for s in list(cases.keys()) if 'RA_label' in s]
        labels_pvira = np.zeros(len(match_labels_pvira))
        for i,_ in enumerate(match_labels_pvira):
            labels_pvira[i] = cases[match_labels_pvira[i]][0]        

        match_labels_pvila = [s for s in list(cases.keys()) if 'LA_label' in s]
        labels_pvila = np.zeros(len(match_labels_pvila))
        for i,_ in enumerate(match_labels_pvila):
            labels_pvila[i] = cases[match_labels_pvila[i]][0] 
                      
        cases.close()        
         
        threshold = self._config['data']['threshold_number']    
            
        train_psd_la, val_psd_la = psd_la[:-threshold], psd_la[-threshold:]
        train_psd_ra, val_psd_ra = psd_ra[:-threshold], psd_ra[-threshold:]

        train_df_la, val_df_la = df_la[:-threshold], df_la[-threshold:]
        train_df_ra, val_df_ra = df_ra[:-threshold], df_ra[-threshold:]

        train_fibre_la, val_fibre_la = fibre_la[:-threshold], fibre_la[-threshold:]
        train_fibre_ra, val_fibre_ra = fibre_ra[:-threshold], fibre_ra[-threshold:]   

        train_mask_la_pvi, val_mask_la_pvi = mask_la_pvi[:-threshold], mask_la_pvi[-threshold:]
        train_mask_ra_pvi, val_mask_ra_pvi = mask_ra_pvi[:-threshold], mask_ra_pvi[-threshold:] 
        train_mask_la_pvi_biatrial, val_mask_la_pvi_biatrial = mask_la_pvi_biatrial[:-threshold], mask_la_pvi_biatrial[-threshold:]
        train_mask_ra_pvi_biatrial, val_mask_ra_pvi_biatrial = mask_ra_pvi_biatrial[:-threshold], mask_ra_pvi_biatrial[-threshold:]  
        train_mask_la_pvira, val_mask_la_pvira = mask_la_pvira[:-threshold], mask_la_pvira[-threshold:]
        train_mask_ra_pvira, val_mask_ra_pvira = mask_ra_pvira[:-threshold], mask_ra_pvira[-threshold:] 
        train_mask_la_pvila, val_mask_la_pvila = mask_la_pvila[:-threshold], mask_la_pvila[-threshold:]
        train_mask_ra_pvila, val_mask_ra_pvila = mask_ra_pvila[:-threshold], mask_ra_pvila[-threshold:] 

        train_labels_pvi, val_labels_pvi = labels_pvi[:-threshold], labels_pvi[-threshold:]                      
        train_labels_pvi_biatrial, val_labels_pvi_biatrial = labels_pvi_biatrial[:-threshold], labels_pvi_biatrial[-threshold:]
        train_labels_pvira, val_labels_pvira = labels_pvira[:-threshold], labels_pvira[-threshold:]
        train_labels_pvila, val_labels_pvila = labels_pvila[:-threshold], labels_pvila[-threshold:]

        set_determinism(seed=0)

        self.train_ds = h5pyDataset(train_labels_pvi, train_labels_pvi_biatrial, \
                                    train_labels_pvira, train_labels_pvila, \
                                    train_psd_la, train_psd_ra, \
                                    train_df_la, train_df_ra, \
                                    train_fibre_la, train_fibre_ra, \
                                    train_mask_la_pvi, train_mask_ra_pvi,train_mask_la_pvi_biatrial, train_mask_ra_pvi_biatrial, \
                                    train_mask_la_pvira, train_mask_ra_pvira,train_mask_la_pvila, train_mask_ra_pvila)
        self.val_ds = h5pyDataset(val_labels_pvi, val_labels_pvi_biatrial, \
                                val_labels_pvira, val_labels_pvila, \
                                val_psd_la, val_psd_ra, \
                                val_df_la, val_df_ra, \
                                val_fibre_la, val_fibre_ra, \
                                val_mask_la_pvi, val_mask_ra_pvi, val_mask_la_pvi_biatrial, val_mask_ra_pvi_biatrial, \
                                val_mask_la_pvira, val_mask_ra_pvira, val_mask_la_pvila, val_mask_ra_pvila)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self._config['training']['batch'],
            shuffle=True,
            num_workers=self._config['hardware']['num_workers'],
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, 
            batch_size=self._config['training']['batch'], 
            num_workers=self._config['hardware']['num_workers'])
        return val_loader

    def configure_optimizers(self):
        if self._config['training']['optimizer']['name'] == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self._config['training']['optimizer']['params']['lr'], 
                weight_decay=self._config['training']['optimizer']['params']['weight_decay'])
        return optimizer

    def training_step(self, batch):
        psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, mask_la, mask_ra, labels = \
            batch['psd_la'].float(), batch['psd_ra'].float(),batch['df_la'].float(), batch['df_ra'].float(), \
            batch['fibre_la'].float(), batch['fibre_ra'].float(), batch['mask_la'].float(), batch['mask_ra'].float(),\
            batch['label']
        output = self.forward(psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, mask_la, mask_ra)
        loss = self.loss_function(output.squeeze(), labels.float())
        self.log("train_loss", loss.item(), sync_dist=True, batch_size=self._config['training']['batch'])
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, mask_la, mask_ra, labels = \
            batch['psd_la'].float(), batch['psd_ra'].float(),batch['df_la'].float(), batch['df_ra'].float(), \
            batch['fibre_la'].float(), batch['fibre_ra'].float(), batch['mask_la'].float(), batch['mask_ra'].float(),\
            batch['label']
        outputs = self.forward(psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, mask_la, mask_ra)
        print(outputs.squeeze(), labels.float(), outputs.squeeze().mean(),labels.float().mean())
        loss = self.loss_function(outputs.squeeze(), labels.float())
        self.log("val_loss", loss.item(), sync_dist=True, batch_size=self._config['training']['batch'])
        tensorboard_logs = {"val_loss": loss.item()}
        self.acc.append(self.accuracy(outputs.squeeze(), labels.float()))
        self.rocauc.append(compute_roc_auc(y_pred=outputs, y=labels))
        self.f1_metrics.append(self.f1_metric(outputs.squeeze().cpu(),labels.cpu()))
        return {"val_loss": loss, "log": tensorboard_logs,"val_number": len(outputs), "predictions": outputs, "labels":labels}
    
    def on_validation_epoch_end(self):        
        mean_val_acc = torch.mean(torch.stack(self.acc))
        mean_val_roc = np.mean(self.rocauc)
        self.rocauc = []
        self.acc = []
        mean_val_f1 = torch.mean(torch.stack(self.f1_metrics))
        self.f1_metrics = []

        tensorboard_logs = {
            "val_roc": mean_val_roc,
            "val_f1": mean_val_f1,
        }
        self.log("val_roc", mean_val_roc, sync_dist=True, batch_size=self._config['training']['batch'])
        self.log("val_acc", mean_val_acc, sync_dist=True, batch_size=self._config['training']['batch'])
        self.log("val_f1", mean_val_f1, sync_dist=True, batch_size=self._config['training']['batch'])

        if mean_val_roc > self.best_val_roc:
            self.best_val_roc = mean_val_roc
            self.best_val_epoch = self.current_epoch

        if mean_val_f1 > self.best_val_f1:
            self.best_val_f1 = mean_val_f1     

        print(
            f"current epoch: {self.current_epoch} "
            f"current mean roc: {mean_val_roc:.4f}"
            f"current mean f1: {mean_val_f1:.4f}"
            f"current mean acc: {mean_val_acc:.4f}"
            f"\nbest mean roc: {self.best_val_roc:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        return {"log": tensorboard_logs}

    