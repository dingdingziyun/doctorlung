from scipy.signal import butter, lfilter
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from flaskexample import app
import wave
import pandas as pd
import numpy as np
import os
import wave
import math
import scipy.io.wavfile as wf
import scipy.signal
import pickle
import librosa
import librosa.display as libd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, ProgbarLogger
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import itertools
import warnings
import seaborn

warnings.filterwarnings("ignore", category=DeprecationWarning)

def read_wav_file(str_filename, target_rate):
	wav = wave.open(str_filename, mode = 'r')
	(sample_rate, data) = extract2FloatArr(wav,str_filename)
	
	if (sample_rate != target_rate):
		( _ , data) = resample(sample_rate, data, target_rate)
		
	wav.close()
	return (target_rate, data.astype(np.float32))

def resample(current_rate, data, target_rate):
	x_original = np.linspace(0,100,len(data))
	x_resampled = np.linspace(0,100, int(len(data) * (target_rate / current_rate)))
	resampled = np.interp(x_resampled, x_original, data)
	return (target_rate, resampled.astype(np.float32))

# -> (sample_rate, data)
def extract2FloatArr(lp_wave, str_filename):
	(bps, channels) = bitrate_channels(lp_wave)
	
	if bps in [1,2,4]:
		(rate, data) = wf.read(str_filename)
		divisor_dict = {1:255, 2:32768}
		if bps in [1,2]:
			divisor = divisor_dict[bps]
			data = np.divide(data, float(divisor)) #clamp to [0.0,1.0]        
		return (rate, data)
	
	elif bps == 3: 
		#24bpp wave
		return read24bitwave(lp_wave)
	
	else:
		raise Exception('Unrecognized wave format: {} bytes per sample'.format(bps))

def read24bitwave(lp_wave):
	nFrames = lp_wave.getnframes()
	buf = lp_wave.readframes(nFrames)
	reshaped = np.frombuffer(buf, np.int8).reshape(nFrames,-1)
	short_output = np.empty((nFrames, 2), dtype = np.int8)
	short_output[:,:] = reshaped[:, -2:]
	short_output = short_output.view(np.int16)
	return (lp_wave.getframerate(), np.divide(short_output, 32768).reshape(-1))

def bitrate_channels(lp_wave):
	bps = (lp_wave.getsampwidth() / lp_wave.getnchannels()) #bytes per sample
	return (bps, lp_wave.getnchannels())

def Freq2Mel(freq):
	return 1125 * np.log(1 + freq / 700)

def Mel2Freq(mel):
	exponents = mel / 1125
	return 700 * (np.exp(exponents) - 1)


#mel_space_freq: the mel frequencies (HZ) of the filter banks, in addition to the two maximum and minimum frequency values
#fft_bin_frequencies: the bin freqencies of the FFT output
#Generates a 2d numpy array, with each row containing each filter bank
def GenerateMelFilterBanks(mel_space_freq, fft_bin_frequencies):
	n_filters = len(mel_space_freq) - 2
	coeff = []
	#Triangular filter windows
	#ripped from http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
	for mel_index in range(n_filters):
		m = int(mel_index + 1)
		filter_bank = []
		for f in fft_bin_frequencies:
			if(f < mel_space_freq[m-1]):
				hm = 0
			elif(f < mel_space_freq[m]):
				hm = (f - mel_space_freq[m-1]) / (mel_space_freq[m] - mel_space_freq[m-1])
			elif(f < mel_space_freq[m + 1]):
				hm = (mel_space_freq[m+1] - f) / (mel_space_freq[m + 1] - mel_space_freq[m])
			else:
				hm = 0
			filter_bank.append(hm)
		coeff.append(filter_bank)
	return np.array(coeff, dtype = np.float32)
 

def FFT2MelSpectrogram(f, Sxx, sample_rate, n_filterbanks):
	(max_mel, min_mel)  = (Freq2Mel(max(f)), Freq2Mel(min(f)))
	mel_bins = np.linspace(min_mel, max_mel, num = (n_filterbanks + 2))
	#Convert mel_bins to corresponding frequencies in hz
	mel_freq = Mel2Freq(mel_bins)
	filter_banks = GenerateMelFilterBanks(mel_freq, f)      
	mel_spectrum = np.matmul(filter_banks, Sxx)
	return (mel_freq[1:-1], np.log10(mel_spectrum  + float(10e-12)))



def sample2MelSpectrum(samples, sample_rate, n_filters):
	n_rows = 175 # 7500 cutoff
	n_window = 512 #~25 ms window
	(f, t, Sxx) = scipy.signal.spectrogram(samples,fs = sample_rate, nfft= n_window, nperseg=n_window)
	Sxx = Sxx[:n_rows,:].astype(np.float32) #sift out coefficients above 7500hz, Sxx has 196 columns
	mel_log = FFT2MelSpectrogram(f[:n_rows], Sxx, sample_rate, n_filters)[1]
	mel_min = np.min(mel_log)
	mel_max = np.max(mel_log)
	diff = mel_max - mel_min
	norm_mel_log = (mel_log - mel_min) / diff if (diff > 0) else np.zeros(shape = (n_filters,Sxx.shape[1]))
	if (diff == 0):
		print('Error: sample data is completely empty')
	
	return (np.reshape(norm_mel_log, (n_filters,Sxx.shape[1])).astype(np.float32)) 

def generate_padded_samples(source, output_length):
	copy = np.zeros(output_length, dtype = np.float32)
	src_length = len(source)
	frac = src_length / output_length
	if(frac < 0.5):
		#tile forward sounds to fill empty space
		cursor = 0
		while(cursor + src_length) < output_length:
			copy[cursor:(cursor + src_length)] = source[:]
			cursor += src_length
	else:
		copy[:src_length] = source[:]
	#
	return copy

##split sound into 5 seconds, and output the padded sound clips
def split_and_pad(original, desiredLength, sampleRate):
	output_buffer_length = int(desiredLength * sampleRate)
	soundclip = original[1]
	n_samples = len(soundclip)
	total_length = n_samples / sampleRate #length of cycle in seconds
	n_slices = int(math.ceil(total_length / desiredLength)) #get the minimum number of slices needed
	samples_per_slice = n_samples // n_slices
	src_start = 0 #Staring index of the samples to copy from the original buffer
	output = [] #Holds the resultant slices
	#print(n_slices)
	for i in range(n_slices):
		src_end = min(src_start + samples_per_slice, n_samples)
		length = src_end - src_start
		copy = generate_padded_samples(soundclip[src_start:src_end], output_buffer_length)
		#clip*times*spectrom
		output.append(copy)
		src_start += length
	return output

# d is the sound clip
def split_and_pad_and_apply_mel_spect(original, desiredLength, sampleRate):
	#number of clips*clip*times*spectrom
	lst_result = split_and_pad(original, desiredLength, sampleRate) 
	freq_result = [sample2MelSpectrum(d, sampleRate, 50) for d in lst_result] #Freq domain
	return freq_result


def extract_wav_features(filename):
	sample_rate = 22000
	n_filters = 50
	n_window = 512
	desired_length = 5
	original = read_wav_file(filename, sample_rate)
	lst_result = split_and_pad(original, desired_length, sample_rate)
	freq_result = [sample2MelSpectrum(d, sample_rate, n_filters) for d in lst_result]
	features = np.array(freq_result)
	#print(np.array(test).shape)
	
	return features

