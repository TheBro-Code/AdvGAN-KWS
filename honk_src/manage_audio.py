from __future__ import print_function
import argparse
import os
import random
import sys
import wave

import librosa
import numpy as np
import pcen
import pyaudio

def set_speech_format(f):
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)

class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_dct_filters=40, n_mels=40, f_max=4000, f_min=20, n_fft=480, hop_ms=10):
        super().__init__()
        self.n_mels = n_mels
        self.dct_filters = np.load('dct_filter.npy')
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = sr // 1000 * hop_ms
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=n_mels, n_fft=n_fft, hop_length=self.hop_length, trainable=True)

    def compute_mfccs(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").astype(np.float32)
        return data

    def compute_pcen(self, data):
        data = self.pcen_transform(data)
        self.pcen_transform.reset()
        return data

    def compute_mel_spectrogram(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        return data

def main():
    parser = argparse.ArgumentParser()
    commands = dict(trim=clean_dir, listen=print_sound_level)
    commands["generate-contrastive"] = generate_dir
    parser.add_argument("subcommand")
    def print_sub_commands():
        print("Subcommands: {}".format(", ".join(commands.keys())))
    if len(sys.argv) <= 1:
        print_sub_commands()
        return
    subcommand = sys.argv[1]
    if subcommand == "generate-contrastive":
        parser.add_argument(
            "directory",
            type=str,
            default=".",
            help="Generate from the directory's audio files")
        flags, _ = parser.parse_known_args()
        generate_dir(flags.directory)
    elif subcommand == "trim":
        parser.add_argument(
            "directory",
            type=str,
            nargs="?",
            default=".",
            help="Trim the directory's audio files")
        flags, _ = parser.parse_known_args()
        clean_dir(flags.directory)
    elif subcommand == "listen":
        print_sound_level()
    else:
        print_sub_commands()

if __name__ == "__main__":
    main()
