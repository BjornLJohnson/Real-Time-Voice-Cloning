import os
from contentEncoder import DeepSpeech as ds
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

_model = None # type: SpeechRecognitionModel
_hparams = None
_device = None
_use_cuda = None

def load_model(model_path=None, save_model_path=None, batch_size=8):
    global _model, _device, _hparams, _use_cuda
    _use_cuda = torch.cuda.is_available()
    _device = torch.device("cuda" if _use_cuda else "cpu")
    warm_start=True

    _hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": 5e-4,
        "batch_size": batch_size,
        "epochs": 10,
        "save_model_path": save_model_path
    }

    _model = ds.SpeechRecognitionModel(
    _hparams['n_cnn_layers'], _hparams['n_rnn_layers'], _hparams['rnn_dim'],
    _hparams['n_class'], _hparams['n_feats'], _hparams['stride'], _hparams['dropout']
    ).to(_device)

    if model_path is not None:
        _model.load_state_dict(torch.load(model_path))

    print(_model)
    print('Num Model Parameters', sum([param.nelement() for param in _model.parameters()]))

def train():
    global _model, _device, _hparams, _use_cuda

    if _model is None:
        raise Exception("Model was not loaded, call load_model() before training")

    dataset_dir = os.path.expanduser("~/dev/datasets/Libri")
    if not os.path.isdir(dataset_dir):
        print("Running")
        os.makedirs(dataset_dir)

    train_url="train-clean-100"
    test_url="test-clean"


    train_dataset = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=train_url, download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=test_url, download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if _use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=_hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: ds.data_processing(x, 'train'),
                                **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=_hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: ds.data_processing(x, 'valid'),
                                **kwargs)


    optimizer = optim.AdamW(_model.parameters(), _hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(_device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=_hparams['learning_rate'], 
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=_hparams['epochs'],
                                            anneal_strategy='linear')

    for epoch in range(1, _hparams['epochs'] + 1):
        ds.train(_model, _device, train_loader, criterion, optimizer, scheduler, epoch)
        ds.test(_model, _device, test_loader, criterion, epoch)
        torch.save(_model.state_dict(), _hparams['save_model_path'])

def generate_text(input_file_path):
    global _model, _device, _hparams, _use_cuda

    if _model is None:
        raise Exception("Model was not loaded, call load_model() before inference")

    waveform, sample_rate = torchaudio.load(input_file_path, normalization=True)
    input_data = [[waveform, None, None, None, None, None]]
    input_layer = ds.data_processing(input_data, 'infer')

    _model.eval()
    output=_model(input_layer[0].to(_device))
    output = F.log_softmax(output, dim=2)
    
    return str(ds.GreedyDecoderInference(output))


