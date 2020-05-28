#!/usr/bin/env python
# coding: utf-8

import os
import DeepSpeech as ds
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data




class ContentEncoder: 
    _model = None
    _hparams = None
    _device = None
    _use_cuda = None
    
    def __init__(self):
        self._use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._use_cuda else "cpu")

        warm_start=True
        
        hparams = {
            "n_cnn_layers": 3,
            "n_rnn_layers": 5,
            "rnn_dim": 512,
            "n_class": 29,
            "n_feats": 128,
            "stride":2,
            "dropout": 0.1,
            "learning_rate": 5e-4,
            "batch_size": 8,
            "epochs": 10
        }
        self._hparams = hparams
        
        self._model = ds.SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(self._device)

        if warm_start :
            self._model.load_state_dict(torch.load("saved_models/deepspeech2.pt"))

        print(self._model)
        print('Num Model Parameters', sum([param.nelement() for param in self._model.parameters()]))
        
    def train(self):
        hparams = self._hparams
        
        if self._model is None:
            raise Exception("Model was not loaded")
            
        dataset_dir = os.path.expanduser("~/dev/datasets/Libri")
        if not os.path.isdir(dataset_dir):
            print("Running")
            os.makedirs(dataset_dir)

        train_url="train-clean-100"
        test_url="test-clean"


        train_dataset = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=train_url, download=True)
        test_dataset = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=test_url, download=True)

        kwargs = {'num_workers': 1, 'pin_memory': True} if self._use_cuda else {}
        train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=hparams['batch_size'],
                                    shuffle=True,
                                    collate_fn=lambda x: ds.data_processing(x, 'train'),
                                    **kwargs)
        test_loader = data.DataLoader(dataset=test_dataset,
                                    batch_size=hparams['batch_size'],
                                    shuffle=False,
                                    collate_fn=lambda x: ds.data_processing(x, 'valid'),
                                    **kwargs)


        optimizer = optim.AdamW(self._model.parameters(), hparams['learning_rate'])
        criterion = nn.CTCLoss(blank=28).to(self._device)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                                steps_per_epoch=int(len(train_loader)),
                                                epochs=hparams['epochs'],
                                                anneal_strategy='linear')

        for epoch in range(1, self._hparams['epochs'] + 1):
            ds.train(self._model, self._device, train_loader, criterion, optimizer, scheduler, epoch)
            ds.test(self._model, self._device, test_loader, criterion, epoch)
            torch.save(self._model.state_dict(), "saved_models/deepspeech3.pt")


