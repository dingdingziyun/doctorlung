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
import itertools
import warnings
import seaborn

def plotwave(putpath, output_path):
	y, sr = librosa.load(putpath)
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
	# Convert to log scale (dB)
	log_S = librosa.power_to_db(S, ref=np.max)
	# Make a new figure
	plt.figure(figsize=(12,4))
	# Display the spectrogram on a mel scale
	# sample rate and hop length parameters are used to render the time axis
	librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
	# draw a color bar
	plt.colorbar(format='%+02.0f dB')
	# Make the figure layout compact
	plt.tight_layout()
	plt.savefig(output_path)
	plt.clf()
	plt.close()
	#plt.gcf().clear()
	#plt_bytes = buf.getvalue()
	return output_path

def plothist(input_df,sounds,sound_per_min, count, output_path):
	crackle_per_min_perpat = pd.read_pickle(input_df)
	crackles_percentile = 100 - round(scipy.stats.percentileofscore(crackle_per_min_perpat[sound_per_min].tolist(),count, kind='strict'), 2)
	crackles_ax = seaborn.distplot(crackle_per_min_perpat[sound_per_min], bins=20, kde=False)
	crackles_ax.set(xlabel=sounds+'/min', ylabel='Patients')
	crackles_ax.title.set_text('Histogram of Number of '+sounds+' in Patients')
	plt.axvline(count, c = 'r', linewidth=4)
	crackles_ax.figure.savefig(output_path)
	plt.clf()
	plt.close()
	return crackles_percentile, output_path

def plotscatter(input_df, n_crackles, n_wheezes, output_path):
	patients_only = pd.read_pickle(input_df)
	seaborn.set(style='ticks')
	_genders= ['COPD','URTI','Bronchiectasis']
	ax = seaborn.lmplot( x="crackles_per_min", y="wheezes_per_min", data=patients_only, fit_reg=False, hue='Diagnosis',hue_order=_genders, legend=True, height=5, aspect=1.5)
	ax.set(xticks=[5, 10, 15, 20, 25, 30], yticks=[5, 10, 15, 20])
	plt.xlabel('Crackles/min', size = 20)
	plt.ylabel('Wheezes/min', size = 20)
	#ax.set(xlabel='Crackles/min', ylabel='Wheezes/min', size = 10)
	#ax.axes.get_yaxis().set_visible(False)
	#ax.set_xticklabels(bars,size = 20)
	plt.axvline(n_crackles, c = 'r', linewidth=2)
	plt.axhline(n_wheezes, c = 'r', linewidth=2)
	plt.savefig(output_path)
	plt.clf()
	plt.close()
	count = patients_only[(patients_only['crackles_per_min'] < n_crackles) & (patients_only['wheezes_per_min'] < n_wheezes)].shape[0]
	together_percentile = round((86-count)*100/86,2)
	return output_path, together_percentile