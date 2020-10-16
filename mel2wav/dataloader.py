### loading training_data


import torch
import torch.utils.data
import torch.nn.functional as F

from librosa.core import load
from librosa.util import normalize

import numpy as np
import random


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    lines = []
    for line in files:
        if int(line.split('|')[4]) < 63.5:
            continue
        lines.append(line.split('|')[0].split('-')[1])
    print('training data length: ', len(lines))
    return lines


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.training_data_dir = os.path.dirname(training_files)
        self.audio_files = training_files
        
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.hop_size = 256
        self.augment = augment

    def __getitem__(self, index):
        # Read audio
        audio_file = os.path.join('{}/audio/audio-{}'.format(self.training_data_dir, self.audio_files[index]))
        mel_file = os.path.join('{}/mels/mel-{}'.format(self.training_data_dir, self.audio_files[index]))
        audio = torch.from_numpy(np.load(audio_file)) 
        mel = torch.from_numpy(np.load(mel_file)).transpose(0, 1)
        mel_length_fixed = self.segment_length // self.hop_size
        # Take segment
        if mel.size(1) <= mel_length_fixed + 1 or wav.size(0) <= self.segment_length:
            wav = torch.nn.functional.pad(wa, (0, self.segment_length - wav.size(0)), 'constant')/data
            return (mel, wav)
          
        max_mel_start = mel.size(1) - mel_length_fixed - 1
        mel_start = random.randint(0, max_mel_start)
        mel = mel[:, mel_start: mel_start + mel_length_fixed]
        wav = wav[mel_start * self.hop_size : (mel_start + mel_length_fixed) * self.hop_size]
        return (mel, wav)


    def __len__(self):
        return len(self.audio_files)
