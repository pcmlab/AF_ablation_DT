

import pytorch_lightning
from monai.utils import set_determinism
from monai.networks.nets import DenseNet121, EfficientNetBN
from monai.data import list_data_collate, DataLoader
import torch
import re
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
from dataset_unified import h5pyDataset
from FOAA import *
from ConvNeXt import * 

class Net(pytorch_lightning.LightningModule):
    def __init__(self, result_dir, **config: Any):
        super().__init__()
        if config['training']['model'] == 'DenseNet121':
            self.model_1 = DenseNet121(spatial_dims=2, in_channels=5, out_channels=config['training']['out_channels'])
            self.model_2 = DenseNet121(spatial_dims=2, in_channels=5, out_channels=config['training']['out_channels'])
        if config['training']['model'] == 'EfficientNetBN':
            self.model_1 = EfficientNetBN("efficientnet-b0", pretrained=False, progress=False, spatial_dims=2, in_channels=5, num_classes=config['training']['out_channels'])
            self.model_2 = EfficientNetBN("efficientnet-b0", pretrained=False, progress=False, spatial_dims=2, in_channels=5, num_classes=config['training']['out_channels'])
        if config['training']['model'] == 'ConvNeXt':
            self.model_1 = ConvNext(out_channels=config['training']['out_channels'])
            self.model_2 = ConvNext(out_channels=config['training']['out_channels'])
        
        
        self.fc = nn.Linear((config['training']['out_channels']+1)*(config['training']['out_channels']+1), config['training']['fc_channels'])
        self.dropout = nn.Dropout(p=config['training']['dropout'])
        self.layer_out = nn.Linear(config['training']['fc_channels'], 4)         #!!! 4 classes
        self.conv_stack= conv_(4,1)   
        self.ln = nn.Linear(config['training']['out_channels']*2, 4) #!!! 4 classes
        self._config = config
        self.result_dir = result_dir
        self.relu = nn.ReLU()     

        #defining layers for FOAA merging method
        self.conv_stack_new= conv_new(config['training']['out_channels'],config['training']['out_channels'])  
        self.fc1 = nn.Linear(config['training']['out_channels'],config['training']['foaa_channels']) 
        self.ln1 = nn.LayerNorm(config['training']['foaa_channels'])
        self.dropout1 = nn.Dropout(p=config['training']['dropout'])
        self.act = nn.ReLU(inplace=True)
        self.layer_out1 = nn.Linear(config['training']['foaa_channels'], 4)

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

    def forward(self, psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, mask_la_pvi, mask_la_pvi_biatrial, mask_ra_pvi, mask_ra_pvi_biatrial):
        
        la = torch.cat((psd_la, df_la, fibre_la, mask_la_pvi, mask_la_pvi_biatrial), dim=1)
        ra = torch.cat((psd_ra, df_ra, fibre_ra, mask_ra_pvi, mask_ra_pvi_biatrial), dim=1)

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
            # x = self.relu(x)
            x = self.layer_out(x)

        if self._config['training']['merging'] == 'concat':
            out_1 = self.model_1(la)
            out_2 = self.model_2(ra)   
            x = torch.cat((out_1, out_2), dim=1)
            x = self.relu(x)
            x = torch.sigmoid(self.ln(x))
            
        if self._config['training']['merging'] == 'FOAA': 
            x1 = self.model_1(la) #run the left abd right atria feature maps through the densenets 
            x3 = self.model_2(ra) # output shape is [80,32]
            #4 outer operations 
            atttention_OA = 'OA'
            atttention_OP = 'OP' 
            atttention_OS = 'OS'
            atttention_OD = 'OD'
            #Create an instance of the the FOAA 

            self.FOAA = FOAA(atttention_OA,atttention_OP,atttention_OS,atttention_OD,x1,x3).to(device="cuda:"+str(self._config['hardware']['gpus'][0]),dtype=torch.float32)
            x = self.FOAA(x1,x3) #call the class method to calculate flattened outer attention 
            x = self.conv_stack_new(x)
            x = x.flatten(start_dim=1)

            x = self.fc1(x) #define all of these layers in pipeline 
            x = self.ln1(x)

            x = self.dropout1(x)
            x = self.act(x)
            x = self.layer_out1(x)

        return x

    def prepare_data(self):
        # set up the correct data path
            
        train_cases = h5py.File(self._config['data']['train_data'] ,'r')

        train_length = self._config['data']['train_data_length']
        val_length = self._config['data']['val_data_length']
        size=self._config['data']['size']

        train_psd_la = np.zeros(((train_length, size, size)))   
        train_psd_ra = np.zeros(((train_length, size, size)))
        train_df_la = np.zeros(((train_length, size, size)))
        train_df_ra = np.zeros(((train_length, size, size)))
        train_fibre_la = np.zeros(((train_length, size, size)))   
        train_fibre_ra = np.zeros(((train_length, size, size)))
        train_mask_la_pvi = np.zeros(((train_length, size, size)))
        train_mask_ra_pvi = np.zeros(((train_length, size, size)))
        train_mask_la_pvi_biatrial = np.zeros(((train_length, size, size)))
        train_mask_ra_pvi_biatrial = np.zeros(((train_length, size, size)))
        train_labels_pvi = np.zeros(train_length)
        train_labels_pvi_biatrial = np.zeros(train_length)
        train_labels_pvira = np.zeros(train_length)
        train_labels_pvila = np.zeros(train_length)

        for s in list(train_cases.keys()):
            ind = int(re.findall(r'\d+', s)[0])-1
            if 'psd_la' in s:
                train_psd_la[ind, :, :] = train_cases[s]
            if 'psd_ra' in s:
                train_psd_ra[ind, :, :] = train_cases[s]
            if 'df_la' in s:
                train_df_la[ind, :, :] = train_cases[s]
            if 'df_ra' in s:
                train_df_ra[ind, :, :] = train_cases[s]
            if 'fibre_la' in s:
                train_fibre_la[ind, :, :] = train_cases[s]
            if 'fibre_ra' in s:
                train_fibre_ra[ind, :, :] = train_cases[s]
            if 'mask_la' in s:
                if 'biatrial' not in s:
                    if 'pvila' not in s:
                        if 'pvira' not in s:
                            train_mask_la_pvi[ind, :, :] = train_cases[s]
            if 'mask_ra' in s:
                if 'biatrial' not in s:
                    if 'pvila' not in s:
                        if 'pvira' not in s:
                            train_mask_ra_pvi[ind, :, :] = train_cases[s]
            if 'mask_la' in s:
                if 'biatrial' in s:
                    if 'pvila' not in s:
                        if 'pvira' not in s:
                            train_mask_la_pvi_biatrial[ind, :, :] = train_cases[s]
            if 'mask_ra' in s:
                if 'biatrial' in s:
                    if 'pvila' not in s:
                        if 'pvira' not in s:
                            train_mask_ra_pvi_biatrial[ind, :, :] = train_cases[s]
            if 'label' in s:
                if 'biatrial' not in s:
                    if 'RA' not in s:
                        if 'LA' not in s:
                            train_labels_pvi[ind] = train_cases[s][0]
            if 'label' in s:
                if 'biatrial' in s:
                    train_labels_pvi_biatrial[ind] = train_cases[s][0]
            if 'RA_label' in s:
                train_labels_pvira[ind] = train_cases[s][0]
            if 'LA_label' in s:
                train_labels_pvila[ind] = train_cases[s][0]
        train_cases.close()

        val_cases = h5py.File(self._config['data']['val_data'] ,'r')

        val_psd_la = np.zeros(((val_length, size, size)))   
        val_psd_ra = np.zeros(((val_length, size, size)))
        val_df_la = np.zeros(((val_length, size, size)))
        val_df_ra = np.zeros(((val_length, size, size)))
        val_fibre_la = np.zeros(((val_length, size, size)))   
        val_fibre_ra = np.zeros(((val_length, size, size)))
        val_mask_la_pvi = np.zeros(((val_length, size, size)))
        val_mask_ra_pvi = np.zeros(((val_length, size, size)))
        val_mask_la_pvi_biatrial = np.zeros(((val_length, size, size)))
        val_mask_ra_pvi_biatrial = np.zeros(((val_length, size, size)))
        val_labels_pvi = np.zeros(val_length)
        val_labels_pvi_biatrial = np.zeros(val_length)
        val_labels_pvira = np.zeros(val_length)
        val_labels_pvila = np.zeros(val_length)

        for s in list(val_cases.keys()):
            ind = int(re.findall(r'\d+', s)[0])-1
            if 'psd_la' in s:
                val_psd_la[ind, :, :] = val_cases[s]
            if 'psd_ra' in s:
                val_psd_ra[ind, :, :] = val_cases[s]
            if 'df_la' in s:
                val_df_la[ind, :, :] = val_cases[s]
            if 'df_ra' in s:
                val_df_ra[ind, :, :] = val_cases[s]
            if 'fibre_la' in s:
                val_fibre_la[ind, :, :] = val_cases[s]
            if 'fibre_ra' in s:
                val_fibre_ra[ind, :, :] = val_cases[s]
            if 'mask_la' in s:
                if 'biatrial' not in s:
                    if 'pvila' not in s:
                        if 'pvira' not in s:
                            val_mask_la_pvi[ind, :, :] = val_cases[s]
            if 'mask_ra' in s:
                if 'biatrial' not in s:
                    if 'pvila' not in s:
                        if 'pvira' not in s:
                            val_mask_ra_pvi[ind, :, :] = val_cases[s]
            if 'mask_la' in s:
                if 'biatrial' in s:
                    if 'pvila' not in s:
                        if 'pvira' not in s:
                            val_mask_la_pvi_biatrial[ind, :, :] = val_cases[s]
            if 'mask_ra' in s:
                if 'biatrial' in s:
                    if 'pvila' not in s:
                        if 'pvira' not in s:
                            val_mask_ra_pvi_biatrial[ind, :, :] = val_cases[s]
            if 'label' in s:
                if 'biatrial' not in s:
                    if 'RA' not in s:
                        if 'LA' not in s:
                            val_labels_pvi[ind] = val_cases[s][0]
            if 'label' in s:
                if 'biatrial' in s:
                    val_labels_pvi_biatrial[ind] = val_cases[s][0]
            if 'RA_label' in s:
                val_labels_pvira[ind] = val_cases[s][0]
            if 'LA_label' in s:
                val_labels_pvila[ind] = val_cases[s][0]
        val_cases.close()


        set_determinism(seed=0)

        self.train_ds = h5pyDataset(train_labels_pvi, train_labels_pvi_biatrial, \
                                    train_labels_pvira, train_labels_pvila, \
                                    train_psd_la, train_psd_ra, \
                                    train_df_la, train_df_ra, \
                                    train_fibre_la, train_fibre_ra, \
                                    train_mask_la_pvi, train_mask_ra_pvi,train_mask_la_pvi_biatrial, train_mask_ra_pvi_biatrial) #, \
                                    #train_mask_la_pvira, train_mask_ra_pvira,train_mask_la_pvila, train_mask_ra_pvila)
        self.val_ds = h5pyDataset(val_labels_pvi, val_labels_pvi_biatrial, \
                                val_labels_pvira, val_labels_pvila, \
                                val_psd_la, val_psd_ra, \
                                val_df_la, val_df_ra, \
                                val_fibre_la, val_fibre_ra, \
                                val_mask_la_pvi, val_mask_ra_pvi, val_mask_la_pvi_biatrial, val_mask_ra_pvi_biatrial) #, \
                                #val_mask_la_pvira, val_mask_ra_pvira, val_mask_la_pvila, val_mask_ra_pvila)

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
        psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, mask_la_pvi, mask_ra_pvi, mask_la_pvi_biatrial, mask_ra_pvi_biatrial, labels = \
            batch['psd_la'].float(), batch['psd_ra'].float(),batch['df_la'].float(), batch['df_ra'].float(), \
            batch['fibre_la'].float(), batch['fibre_ra'].float(), batch['mask_la_pvi'].float(), batch['mask_ra_pvi'].float(),\
            batch['mask_la_pvi_biatrial'].float(), batch['mask_ra_pvi_biatrial'].float(), batch['labels']
        output = self.forward(psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, mask_la_pvi, mask_la_pvi_biatrial, mask_ra_pvi, mask_ra_pvi_biatrial)
        loss = self.loss_function(output.squeeze(), labels.float())
        self.log("train_loss", loss.item(), sync_dist=True, batch_size=self._config['training']['batch'])
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, mask_la_pvi, mask_ra_pvi, mask_la_pvi_biatrial, mask_ra_pvi_biatrial, labels = \
            batch['psd_la'].float(), batch['psd_ra'].float(),batch['df_la'].float(), batch['df_ra'].float(), \
            batch['fibre_la'].float(), batch['fibre_ra'].float(), batch['mask_la_pvi'].float(), batch['mask_ra_pvi'].float(),\
            batch['mask_la_pvi_biatrial'].float(), batch['mask_ra_pvi_biatrial'].float(), batch['labels']
        outputs = self.forward(psd_la, psd_ra, df_la, df_ra, fibre_la, fibre_ra, mask_la_pvi, mask_la_pvi_biatrial, mask_ra_pvi, mask_ra_pvi_biatrial)
        loss = self.loss_function(outputs.squeeze(), labels)
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

    
