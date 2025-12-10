import wave
import scipy.fft
import sounddevice as sd
import numpy as np
from playsound import playsound
from scipy.io.wavfile import write, read

fs = 44100 # sample rate
duration = 3
print("Recording....")
recording = sd.rec(int(duration*fs),samplerate=fs,channels=1)
sd.wait() #Wait for recording to be finished
write('output.wav', fs, recording)
print("Done!")
print("Press Enter to Play Audio")

 ---------------------
plt.figure(figsize=(10, 4))
plt.plot(t, data)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Recorded audio waveform")
plt.tight_layout()
plt.show()
plt.specgram(data, Fs=fs_loaded, NFFT=1024, noverlap=512)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram")
plt.colorbar(label="Intensity (dB)")
plt.show()

playsound('your\path\here\output.wav')
fs_loaded, data = read("output.wav")
import sounddevice as sd
import scipy.fft
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt
import numpy as np
fs_loaded, data = read("output.wav")

# If stereo, convert to mono for plotting:
if len(data.shape) > 1:
    data = data[:, 0]

# Time axis for plotting
t = np.linspace(0, len(data)/fs_loaded, num=len(data))

# ---------------------
# 3. Plot
# ---------------------
plt.figure(figsize=(10, 4))
plt.plot(t, data)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Recorded audio waveform")
plt.tight_layout()
plt.show()
plt.specgram(data, Fs=fs_loaded, NFFT=1024, noverlap=512)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram")
plt.colorbar(label="Intensity (dB)")
plt.show()

N=len(data)
yf=np.fft.fft(data)
xf=np.fft.fftfreq(N,1/fs)
yf_shifted=np.fft.fftshift(yf)
xf_shifted=np.fft.fftshift(xf)
Mag=np.abs(yf_shifted)
MagMax = max(Mag)
MagSca= Mag/MagMax
plt.figure(figsize=(10,6))
plt.plot(xf_shifted,MagSca)
plt.xlim(0,4000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Analysis from Recorded Data")

import sounddevice as sd
import scipy.fft
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

n=int(len(xf_shifted+1)/2)

xfp=xf_shifted[n:]
Magp=Mag[n:]
inds=find_peaks(Magp,height=.2,width=10)
peakfreqs=xfp[inds[0][np.argsort(inds[1]["peak_heights"])]]
peakmags=Magp[inds[0][np.argsort(inds[1]["peak_heights"])]]

n=int(input("How Many Frequencies Do you want to pick?"))
playsound(Your\Path\Here\output.wav')
Fourier=peakfreqs[-n:]
print(Fourier)
MagR=peakmags[-n:]
MagRMAX=max(MagR)
ovrscale=1
MagRSCALED=ovrscale*(MagR/MagRMAX)
print("Here are the picked frequencies")
print(MagRSCALED)
finalsig=np.zeros(len(t))
for i in range(len(Fourier)):
    wave = MagRSCALED[i]*np.sin(2*(np.pi)*Fourier[i]*t)
finalsig
    
plt.plot(t,finalsig,label="Combined Waveform")
tmax=.01
plt.xlim(0,tmax)
plt.xlabel("Time (s)")
plt.ylabel("Scaled Amplitude")
plt.legend()
plt.show()
sd.play(finalsig)