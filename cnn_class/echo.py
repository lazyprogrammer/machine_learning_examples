# https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow
# https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

from scipy.io.wavfile import write

# If you right-click on the file and go to "Get Info", you can see:
# sampling rate = 16000 Hz
# bits per sample = 16
# The first is quantization in time
# The second is quantization in amplitude
# We also do this for images!
# 2^16 = 65536 is how many different sound levels we have
# 2^8 * 2^8 * 2^8 = 2^24 is how many different colors we can represent

spf = wave.open('helloworld.wav', 'r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
print("numpy signal:", signal.shape)

plt.plot(signal)
plt.title("Hello world without echo")
plt.show()

delta = np.array([1., 0., 0.])
noecho = np.convolve(signal, delta)
print("noecho signal:", noecho.shape)
assert(np.abs(noecho[:len(signal)] - signal).sum() < 0.000001)

noecho = noecho.astype(np.int16) # make sure you do this, otherwise, you will get VERY LOUD NOISE
write('noecho.wav', 16000, noecho)

filt = np.zeros(16000)
filt[0] = 1
filt[4000] = 0.6
filt[8000] = 0.3
filt[12000] = 0.2
filt[15999] = 0.1
out = np.convolve(signal, filt)

out = out.astype(np.int16) # make sure you do this, otherwise, you will get VERY LOUD NOISE
write('out.wav', 16000, out)

# plt.plot(out)
# plt.title("Hello world with small echo")
# plt.show()


