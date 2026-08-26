import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sp
from scipy.signal import butter, filtfilt

video_path = "/home/davig/code/pulseIQ/pulseiq/WhatsApp Video 2026-08-26 at 15.13.21.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Capture FPS:", fps)

red = []
count = 0

temp = []

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    red_chanel = frame[:, :, 2]

    brightness = float(np.mean(red_chanel))


    red.append(brightness)
    count += 1

cap.release()

red = np.array(red)
b, a = butter(3, [0.7, 4], btype="band", fs=fps)
ppg = filtfilt(b, a, red - red.mean())

freqs = np.fft.rfftfreq(len(ppg), 1 / fps)
spec = np.abs(np.fft.rfft(ppg))
m = (freqs >= 0.7) & (freqs <= 4)

print(f"Pulse: {freqs[m][spec[m].argmax()] * 60:.0f} beats/minute")


df = pd.DataFrame()

df['Pulse'] = ppg[:1000]


plt.plot(df.index, df["Pulse"])
plt.show()
