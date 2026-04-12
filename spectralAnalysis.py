import numpy as np
import scipy as sp

# Sampling window
def getWindow(windowType, n):
    if windowType == "rect":
        return np.ones(n)
    return sp.signal.get_window(windowType, n)

# Correlogram & Periodogram
def getACF(x, maxLag=None, biased=True, autoCovariance=True):
    if autoCovariance:
        x = x - np.mean(x)

    N = len(x)

    X = np.fft.rfft(x, n=2 * N)
    r = np.fft.irfft(np.abs(X)**2, n=2 * N)[:N]

    if biased:
        r /= N
    else:
        r /= np.arange(N, 0, -1)

    if maxLag is not None:
        r = r[:maxLag + 1]
    return r

def getCorrelogram(x, samplingRate):
    r = getACF(x)
    r = np.concatenate((r[1:][::-1], r))
    r = np.fft.ifftshift(r)

    correlogram = np.real(np.fft.rfft(r) / samplingRate)
    frequencies = np.fft.rfftfreq(len(r), d=1 / samplingRate)

    correlogram[1:] *= 2
    return np.real(correlogram), frequencies

def getPeriodogram(x, samplingRate):
    signalLength = len(x)
    x = x - np.mean(x)

    X = np.fft.rfft(x)
    periodogram = np.abs(X)**2 / signalLength / samplingRate
    frequencies = np.fft.rfftfreq(signalLength, d=1 / samplingRate)

    if signalLength % 2 == 0:
        periodogram[1:-1] *= 2  # Exclude 0 and pi
    else:
        periodogram[1:] *= 2    # Exclude 0

    return periodogram, frequencies

# PSD
def getBlackmanTukey(x, samplingRate, windowType="rect", windowLength=50001):
    r = getACF(x, maxLag=windowLength // 2)
    r = np.concatenate((r[1:][::-1], r))

    w = getWindow(windowType, len(r))
    r = np.fft.ifftshift(r * w)
    psd = np.real(np.fft.rfft(r)) / samplingRate
    frequencies = np.fft.rfftfreq(len(r), d=1 / samplingRate)

    psd[1:] *= 2
    return psd, frequencies

def getBartlett(x, samplingRate, numSegment=10):
    segmentLength = len(x) // numSegment
    x = x[:segmentLength * numSegment]

    frequencies = np.fft.rfftfreq(segmentLength, d=1 / samplingRate)
    psd = np.zeros(len(frequencies))

    for i in range(numSegment):
        segment = x[i * segmentLength:(i + 1) * segmentLength]
        segment = segment - np.mean(segment)
        X = np.fft.rfft(segment)
        periodogram = (np.abs(X)**2) / segmentLength / samplingRate
        if segmentLength % 2 == 0:
            periodogram[1:-1] *= 2
        else:
            periodogram[1:] *= 2
        psd += periodogram

    psd /= numSegment
    return psd, frequencies

def getWelch(x, samplingRate, numSegment=10, windowType="rect", overlap=0.5):
    signalLength = len(x)
    segmentLength = int(signalLength / (1 + (numSegment - 1) * (1 - overlap)))
    overlapLength = int(segmentLength * overlap)
    stepSize = segmentLength - overlapLength

    window = getWindow(windowType, segmentLength)
    windowPower = np.sum(window**2)
    frequencies = np.fft.rfftfreq(segmentLength, d=1 / samplingRate)
    psd = np.zeros(len(frequencies))

    for i in range(numSegment):
        i0 = i * stepSize
        segment = x[i0:i0 + segmentLength]

        if len(segment) < segmentLength:
            numSegment = i
            break

        segment = segment - np.mean(segment)
        X = np.fft.rfft(window * segment)
        periodogram = (np.abs(X)**2) / windowPower / samplingRate
        if segmentLength % 2 == 0:
            periodogram[1:-1] *= 2
        else:
            periodogram[1:] *=2
        psd += periodogram

    if numSegment == 0:
        raise ValueError("No complete segments are available")

    psd /= numSegment
    return psd, frequencies

# STFT
def getSpectrogram(x, samplingRate, M, windowType="hamming"):
    window = getWindow(windowType, M)

    step = M // 4
    samples = np.arange(0, len(x) - M, step)
    spectrogram = np.zeros((M // 2 + 1, len(samples)))
    for idx, k in enumerate(samples):
        windowed = x[k:k + M] * window
        X = np.fft.rfft(windowed)
        spectrogram[:, idx] = 20 * np.log10(np.maximum(np.abs(X), 1e-6))    # Amplitude

    # frequencies = np.linspace(0, samplingRate // 2, M // 2 + 1)
    frequencies = np.fft.rfftfreq(M, d=1 / samplingRate)
    times = samples / samplingRate
    return spectrogram, times, frequencies

# Cross-correlation
def getCCF(s1, s2, samplingRate, biased=True):
    assert len(s1) == len(s2)
    N = len(s1)
    S1 = np.fft.rfft(s1, n=2 * N)
    S2 = np.fft.rfft(s2, n=2 * N)

    crossCorrelation = S1 * np.conjugate(S2)
    crossCorrelation = np.fft.irfft(crossCorrelation, n=2 * N)
    crossCorrelation = np.fft.ifftshift(crossCorrelation)[1:]

    if biased:
        crossCorrelation /= N
    else:
        crossCorrelation /= np.concatenate([np.arange(1, N + 1), np.arange(N - 1, 0, -1)])

    offset = np.arange(-(N - 1), N) / samplingRate
    return crossCorrelation, offset

def getGCC(s1, s2, samplingRate, weighting="phat", eps=1e-12):
    assert len(s1) == len(s2)
    N = len(s1)
    S1 = np.fft.rfft(s1, n=2 * N)
    S2 = np.fft.rfft(s2, n=2 * N)

    G = S1 * np.conjugate(S2)

    if weighting == "phat":
        G /= np.abs(G) + eps
    elif weighting == "scot":
        P1 = np.abs(S1)**2
        P2 = np.abs(S2)**2
        G /= np.sqrt(P1 * P2) + eps
    elif weighting == "roth":
        P1 = np.abs(S1) ** 2
        G /= P1 + eps
    else: pass  # CCF

    crossCorrelation = np.fft.irfft(G, n=2 * N)
    crossCorrelation = np.fft.ifftshift(crossCorrelation)[1:]

    offset = np.arange(-(N - 1), N) / samplingRate
    return crossCorrelation, offset

# Environment
def addDelay(x, samplingRate, delay=2):
    delaySamples = int(samplingRate * delay)
    # return np.roll(x, int(samplingRate * delay))
    return np.concatenate((np.zeros(delaySamples, dtype=x.dtype), x[:-delaySamples]))

def addAWGN(x, snr):
    signalPower = np.mean(x**2)
    noisePower = signalPower / 10**(snr / 10)
    noise = np.sqrt(noisePower) * np.random.randn(len(x))
    return x + noise

def addColoredNoise(x, snr, noiseType="lowpass"):
    white = np.random.randn(len(x))

    if noiseType == "lowpass" or noiseType == "highpass":
        b, a = sp.signal.butter(4, 0.15, btype=noiseType)
        colored = sp.signal.lfilter(b, a, white)
    else:
        colored = white

    colored /= np.std(colored) + 1e-12

    signalPower = np.mean(x**2)
    noisePower = signalPower / 10**(snr / 10)
    colored *= np.sqrt(noisePower)

    return x + colored

def addEcho(x, samplingRate, delay, attenuation):
    delaySamples = int(samplingRate * delay)

    if not (0 < delaySamples < len(x)):
        raise ValueError(f"delaySamples out of range (0, {len(x)})")

    y = np.copy(x)
    y[delaySamples:] += attenuation * x[:-delaySamples]
    return y

def addMultipath(x, samplingRate, delay, gain):
    if len(delay) != len(gain):
        raise ValueError("delay and gain must have the same length")

    delaySamples = [int(d * samplingRate) for d in delay]

    y = np.zeros_like(x)
    for d, g in zip(delaySamples, gain):
        if d == 0:
            y += g * x
        elif 0 < d < len(x):
            y[d:] += g * x[:-d]
    return y
