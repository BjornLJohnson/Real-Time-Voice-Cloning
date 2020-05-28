import sys
import os
import logging
import subprocess
# import shlex
import numpy as np
import glob
import webrtcvad
from contentEncoder import wavSplit
from deepspeech import Model

_model = None # type: 
_device = None # type: torch.device


'''
Load the pre-trained model into the memory
@param models: Output Grapgh Protocol Buffer file
@param scorer: Scorer file
@Retval
Returns a list [DeepSpeech Object, Model Load Time, Scorer Load Time]
'''
def load_model(modelDir):
    # Point to a path containing the pre-trained models & resolve ~ if used
    dirName = os.path.expanduser(modelDir)

    # Resolve all the paths of model files
    output_graph = glob.glob(dirName + "/*.pbmm")[0]
    logging.debug("Found Model: %s" % output_graph)

    scorer = glob.glob(dirName + "/*.scorer")[0]
    logging.debug("Found scorer: %s" % scorer)

    # model_retval = load_model(output_graph, scorer)

    _model = Model(output_graph)
    _model.enableExternalScorer(scorer)
    print("Loaded DeepSpeech transcriber")
    
    
def is_loaded():
    return _model is not None


'''
Run Inference on input audio file
@param ds: Deepspeech object
@param audio: Input audio for running inference on
@param fs: Sample rate of the input audio file
@Retval:
Returns a list [Inference, Inference Time, Audio Length]
'''
def transcribe_audio(audio, fs):
    audio_length = len(audio) * (1 / fs)

    # Run Deepspeech
    output = _model.stt(audio)

    return output


'''
Generate VAD segments. Filters out non-voiced audio frames.
@param waveFile: Input wav file to run VAD on.0
@Retval:
Returns tuple of
    segments: a bytearray of multiple smaller audio frames
              (The longer audio split into mutiple smaller one's)
    sample_rate: Sample rate of the input audio file
    audio_length: Duraton of the input audio file
'''
def vad_segment_generator(wavFile, aggressiveness):
    logging.debug("Caught the wav file @: %s" % (wavFile))
    audio, sample_rate, audio_length = wavSplit.read_wave(wavFile)
    assert sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = wavSplit.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = wavSplit.vad_collector(sample_rate, 30, 300, vad, frames)

    return segments, sample_rate, audio_length