# Speech-To-Speech-Synthesis

## Set Up:
* Clone using -r flag to initialize submodule
* Create virtual environment w/ Python 3.7 (3.8 does NOT work)
* Install requirements (Note: I was using my mac and could not get tensorflow-gpu, try tensorflow==1.14 if not working)
    ```
    pip install requirements.txt
    ```
* Install pytorch (needs to be CUDA enabled)
* Download [pretrained models] and place "saved_models" folders into synthesizer/encoder/vocoder directories
* Download [LibriSpeech Dataset](http://www.openslr.org/resources/12/train-clean-100.tar.gz), unzip into <datasets_root>
* Run toolbox script
    ```python
    python demo_toolbox.py -d <datasets_root>
    ```
* Record a segment of your own voice and generate new speech from text

[pretrained models]:https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models
Real-Time-Voice-Cloning/README.md
