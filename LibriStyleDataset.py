import os
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    unicode_csv_reader,
    walk_files,
)
from encoder import inference as styleEncoder

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"


def load_libristyle_item(fileid, path, ext_audio, ext_embed):

    speaker_id, chapter_id, utterance_id = fileid.split("-")

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
    file_embed = fileid_audio + ext_embed
    file_embed = os.path.join(path, speaker_id, chapter_id, file_embed)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)
    
    embed = loadtxt(file_embed, delimiter=',')

    return (
        waveform,
        embed
    )


class LibriStyle(Dataset):
    """
    Create a Dataset for LibriSpeech. Each item is a tuple of the form:
    waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"
    _ext_embed = ".csv"

    def __init__(
        self, root, url=URL, folder_in_archive=FOLDER_IN_ARCHIVE, download=False, preprocess=False
    ):

        if url in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:

            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/resources/12/"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url(url, root)
                extract_archive(archive)

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)
        
        if preprocess:
            self.preprocess_embeddings(self._path, self._ext_audio, self._ext_embed)

    def __getitem__(self, n):
        fileid = self._walker[n]
        return load_libristyle_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __len__(self):
        return len(self._walker)
    
    def preprocess_embeddings(self, path, ext_audio, ext_embed):
        for i in range(1, len(self._walker)):
            fileid = self._walker[i]
            
            speaker_id, chapter_id, utterance_id = fileid.split("-")

            fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
            file_audio = fileid_audio + ext_audio
            file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
            file_embed = fileid_audio + ext_embed
            file_embed = os.path.join(path, speaker_id, chapter_id, file_embed)

            # Load audio
            waveform, sample_rate = torchaudio.load(file_audio)
            
            print("Loaded file: ", fileid)
            
            # Calculate speaker embedding
            wav = waveform.transpose(0,1).detach().numpy().squeeze()
            preprocessed_wav = styleEncoder.preprocess_wav(wav, sample_rate)
            embedding = styleEncoder.embed_utterance(preprocessed_wav)

            # Save embeddings to corresponding csv files
            data = asarray(embedding)
            savetxt(file_embed, data, delimiter=',')
            
            print("Saved embedding: ", file_embed)