from typing import Optional
import threading
import collections

import numpy as np
import scipy as sp
import sounddevice as sd

from spectralAnalysis import *

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


RATE = 44100
DEVICE = 1
OUTPUT_DEVICE = 0
CHANNELS = 1
BLOCK_SIZE = 1024
MAX_FREQUENCY = 12000

SPECTRUM_WINDOW_TYPE = "hann"
SPECTRUM_METHOD = "welch"
SPECTRUM_NUM_SEGMENTS = 16
SPECTRUM_OVERLAP = 0.5
SPECTRUM_REFRESH_RATE = 5
SPECTRUM_FORMAT = "psd"
SPECTRUM_FORMATS = ("psd", "power", "linear")
SPECTRUM_DISPLAY_MODE = "linear"
SPECTRUM_ALPHA = 0
SPECTRUM_DEBUG = False

SPECTRUM_DB_LIM = (-120, 0)
SPECTRUM_LINEAR_LIM = {
    "psd": (-1e-6, 5e-5),
    "power": (-1e-6, 5e-4),
    "linear": (-1e-6, 0.025),
}

WAVEFORM_ENVELOPE = True
WAVEFORM_PLOT_POINTS = 256

SPECTROGRAM_WINDOW_TYPE = "hann"
SPECTROGRAM_OVERLAP = 0.5
SPECTROGRAM_REFRESH_RATE = 5

WAVEFORM_DURATION = 0.15
SPECTRUM_DURATION = 0.5
BUFFER_DURATION = 3.0
SNAPSHOT_DURATION = 3.0

CONFLICT_KEYMAPS = [
    "keymap.save",
    "keymap.yscale",
    "keymap.pan",
    "keymap.home",
    "keymap.back",
    "keymap.zoom",
]


class realtimeTracking:
    def __init__(self,
                 samplingRate: Optional[int] = RATE,
                 waveformDuration: Optional[float] = WAVEFORM_DURATION,
                 spectrumDuration: Optional[float] = SPECTRUM_DURATION,
                 bufferDuration: Optional[float] = BUFFER_DURATION,
                 snapshotDuration: Optional[float] = SNAPSHOT_DURATION,
                 spectrumWindowType: Optional[str] = SPECTRUM_WINDOW_TYPE,
                 spectrumMethod: Optional[str] = SPECTRUM_METHOD,
                 nSpectrumSegments: Optional[int] = SPECTRUM_NUM_SEGMENTS,
                 spectrumOverlap: Optional[float] = SPECTRUM_OVERLAP,
                 spectrumRefreshRate: Optional[int] = SPECTRUM_REFRESH_RATE,
                 spectrumFormat: Optional[str] = SPECTRUM_FORMAT,
                 spectrumDisplayMode: Optional[str] = SPECTRUM_DISPLAY_MODE,
                 spectrogramWindowType: Optional[str] = SPECTROGRAM_WINDOW_TYPE,
                 spectrogramOverlap: Optional[float] = SPECTROGRAM_OVERLAP,
                 spectrogramRefreshRate: Optional[int] = SPECTROGRAM_REFRESH_RATE,
                 device: Optional[int] = DEVICE,
                 outputDevice: Optional[int] = OUTPUT_DEVICE,
                 channels: Optional[int] = CHANNELS,
                 blockSize: Optional[int] = BLOCK_SIZE):
        self.samplingRate = samplingRate
        self.device = device
        self.outputDevice = outputDevice
        self.channels = channels
        self.blockSize = blockSize

        self.waveformDuration = waveformDuration
        self.spectrumDuration = spectrumDuration
        self.bufferDuration = bufferDuration
        self.snapshotDuration = snapshotDuration

        self.nWaveformSamples = int(samplingRate * self.waveformDuration)
        self.nSpectrumSamples = int(samplingRate * self.spectrumDuration)
        self.nBufferSamples = int(samplingRate * self.bufferDuration)
        self.nSnapshotSamples = int(samplingRate * self.snapshotDuration)

        if self.nSnapshotSamples > self.nBufferSamples:
            raise ValueError("Snapshot duration cannot exceed buffer length.")

        self.buffer = np.zeros(self.nBufferSamples, dtype=np.float32)
        self.writePos = 0
        self.filledSamples = 0

        self.inputStreamLock = threading.Lock()
        self.outputStreamLock = threading.Lock()
        self.snapshotLock = threading.Lock()
        self.bufferLock = threading.Lock()
        self.playbackLock = threading.Lock()

        self._callbackStatus = collections.deque(maxlen=32)
        self._pendingPlayback = None
        self._activePlayback = None

        self.captureEnabled = threading.Event()
        self.captureEnabled.set()
        self.closingEvent = threading.Event()

        self.snapshotSamples = None
        self.snapshotWave = None
        self.snapshotProcessed = None
        self.snapshotSpectrumSamples = None
        self.snapshotSpectrumFreq = None
        self.snapshotSpectrum = None

        self.spectrumWindowType = spectrumWindowType
        self.spectrumMethod = spectrumMethod.strip().lower()
        self.nSpectrumSegments = nSpectrumSegments
        self.spectrumOverlap = spectrumOverlap
        self.spectrumRefreshRate = spectrumRefreshRate
        self.spectrumFormat = spectrumFormat.strip().lower()
        self.spectrumDisplayMode = spectrumDisplayMode.strip().lower()

        if self.spectrumFormat not in SPECTRUM_FORMATS:
            raise ValueError("Invalid spectrum format")
        if self.spectrumDisplayMode not in ("linear", "db"):
            raise ValueError("Invalid spectrum display mode")
        if self.spectrumMethod not in ("correlogram", "periodogram", "blackman-tukey", "bartlett", "welch"):
            raise ValueError("Invalid spectral analysis method")
        
        if self.spectrumMethod == "blackman-tukey":
            self.spectrumWindowLength = max(self.blockSize, self.nSpectrumSamples // 2)
            self.spectrumSnapshotWindowLength = max(self.blockSize, self.nSnapshotSamples // 2)

            self.spectrumWindowLength -= (self.spectrumWindowLength % 2 == 0)
            self.spectrumSnapshotWindowLength -= (self.spectrumSnapshotWindowLength % 2 == 0)
        elif self.spectrumMethod == "bartlett":
            self.spectrumWindowLength = max(self.blockSize, self.nSpectrumSamples // self.nSpectrumSegments)
            self.spectrumSnapshotWindowLength = max(self.blockSize, self.nSnapshotSamples // self.nSpectrumSegments)
        elif self.spectrumMethod == "welch":
            self.spectrumWindowLength = max(self.blockSize, self.nSpectrumSamples // int(1 + (self.nSpectrumSegments - 1) * (1 - self.spectrumOverlap)))
            self.spectrumSnapshotWindowLength = max(self.blockSize, self.nSnapshotSamples // int(1 + (self.nSpectrumSegments - 1) * (1 - self.spectrumOverlap)))
        else:
            self.spectrumWindowLength = self.nSpectrumSamples
            self.spectrumSnapshotWindowLength = self.nSnapshotSamples

        self.spectrogramWindowType = spectrogramWindowType
        self.spectrogramWindowLength = max(self.blockSize, self.nSnapshotSamples // 128)
        self.spectrogramSnapshotWindowLength = max(self.blockSize, self.nSnapshotSamples // 128)
        self.spectrogramOverlap = spectrogramOverlap
        self.spectrogramRefreshRate = spectrogramRefreshRate

        self.processFilter = sp.signal.butter(2, [100, 2500], btype="bandpass", fs=self.samplingRate, output="sos")

        self.inputStream = sd.InputStream(
            samplerate=self.samplingRate,
            channels=self.channels,
            blocksize=self.blockSize,
            dtype="float32",
            device=self.device,
            callback=self.audioCallback,
        )
        self.outputStream = sd.OutputStream(
            samplerate=self.samplingRate,
            channels=self.channels,
            blocksize=self.blockSize,
            dtype="float32",
            device=self.outputDevice,
            callback=self.outputCallback,
        )

        self.liveFig, (self.axWave, self.axSpectrum, self.axSpectrogram) = plt.subplots(3, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [1, 3, 2]})
        self.ani = None

        if WAVEFORM_PLOT_POINTS > 0:
            self.waveStride = self.nWaveformSamples // WAVEFORM_PLOT_POINTS
        else:
            self.waveStride = 1

        if WAVEFORM_PLOT_POINTS == 0:
            self.waveTime = np.arange(self.nWaveformSamples) / self.samplingRate
        elif WAVEFORM_ENVELOPE:
            self.waveTime = np.repeat(np.arange(self.nWaveformSamples // self.waveStride) * self.waveStride / self.samplingRate, 2)
        else:
            self.waveTime = np.arange(0, self.nWaveformSamples, self.waveStride) / self.samplingRate
        self.liveSpectrum = np.zeros(0, dtype=np.float32)
        self.liveSpectrumFreq = np.zeros(0, dtype=np.float32)
        self.livePSD = np.zeros(0, dtype=np.float32)
        self.liveSpectrogram, self.liveSpectrogramTimes, self.liveSpectrogramFreq = self.estSpectrogram(np.zeros(self.nSpectrumSamples, dtype=np.float32), self.spectrogramWindowLength)

        self.frameIndex = 0

        self.plotWave, = self.axWave.plot(self.waveTime, np.zeros_like(self.waveTime))
        self.axWave.set_title("Live waveform")
        self.axWave.set_xlabel("Time (s)")
        self.axWave.set_ylabel("Amplitude")
        self.axWave.set_xlim(0, self.waveformDuration)
        self.axWave.set_ylim(-1, 1)
        self.axWave.grid(True)
        self.statusLine = self.axWave.text(0.985, 0.95, "test", transform=self.axWave.transAxes, ha="right", va="top",
                                           horizontalalignment="center", verticalalignment="bottom", fontsize=9,
                                           bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},)

        self.plotSpectrum, = self.axSpectrum.plot(self.liveSpectrumFreq, self.liveSpectrum)
        self.axSpectrum.set_title(f"Live spectrum - {self.spectrumMethod.upper()} / {self.spectrumWindowType.upper()}")
        self.axSpectrum.set_xlabel("Frequency (Hz)")
        self.axSpectrum.set_xlim(20, MAX_FREQUENCY)
        self._applySpectrumStyle(self.axSpectrum)
        self.axSpectrum.grid(True)

        self.plotSpectrogram = self.axSpectrogram.imshow(self.liveSpectrogram,
                                                         origin="lower", aspect="auto", cmap="turbo",
                                                         extent=(0, self.spectrumDuration, 0, MAX_FREQUENCY),
                                                         interpolation="nearest")
        self.axSpectrogram.set_title(f"Live spectrogram - {self.spectrumMethod.upper()} / {self.spectrumWindowType.upper()}")
        self.axSpectrogram.set_xlabel("Time (s)")
        self.axSpectrogram.set_ylabel("Frequency (Hz)")
        self.plotSpectrogram.set_clim(-120, 0)

        self.snapshotFig, self.snapshotAxs = None, None
        self.snapshotWaveRaw = None
        self.snapshotWaveProcessed = None
        self.snapshotSpectrumRaw = None
        self.snapshotSpectrumProcessed = None
        self.snapshotSpectrogramRaw = None
        self.snapshotSpectrogramProcessed = None

    def audioCallback(self, indata, frames, time, status):
        if status:
            self._callbackStatus.append(str(status))

        if not self.captureEnabled.is_set(): return

        x = indata[:, 0]
        signalLength = len(x)

        with self.bufferLock:
            endPos = self.writePos + signalLength
            if endPos <= self.nBufferSamples:
                self.buffer[self.writePos:endPos] = x
            else:
                i0 = self.nBufferSamples - self.writePos
                self.buffer[self.writePos:] = x[:i0]
                self.buffer[:endPos % self.nBufferSamples] = x[i0:]

            self.writePos = endPos % self.nBufferSamples
            self.filledSamples = min(self.nBufferSamples, self.filledSamples + signalLength)

    def outputCallback(self, outdata, frames, time, status):
        if status:
            self._callbackStatus.append(str(status))

        with self.playbackLock:
            if self._pendingPlayback is not None:
                self._activePlayback = self._pendingPlayback
                self._pendingPlayback = None

        outdata[:, 0].fill(0.0)
        if self._activePlayback is not None:
            buf, pos = self._activePlayback
            count = min(frames, len(buf) - pos)
            if count > 0:
                outdata[:count, 0] = buf[pos:pos + count]
                pos += count
            self._activePlayback = (buf, pos) if pos < len(buf) else None

    def getSamples(self, nSamples, pad=True):
        available = min(nSamples, self.filledSamples)
        if pad:
            samples = np.zeros(nSamples, dtype=np.float32)
        else:
            if available == 0:
                return np.zeros(0, dtype=np.float32)
            samples = np.empty(available, dtype=np.float32)

        if available == 0:
            return samples

        i0 = (self.writePos - available) % self.nBufferSamples
        endPos = i0 + available

        if endPos <= self.nBufferSamples:
            samples[-available:] = self.buffer[i0:endPos]
        else:
            tmp = self.nBufferSamples - i0
            samples[-available:-available + tmp] = self.buffer[i0:]
            samples[-available + tmp:] = self.buffer[:available - tmp]

        return samples

    def estSpectrum(self, x, windowLength):
        psd, frequencies = self._estPSD(x, windowLength)
        return self._formatSpectrum(psd, frequencies), frequencies

    def _estPSD(self, x, windowLength):
        method = self.spectrumMethod

        if method == "correlogram":
            return getCorrelogram(x, self.samplingRate)
        if method == "periodogram":
            return getPeriodogram(x, self.samplingRate)
        if method == "blackman-tukey":
            return getBlackmanTukey(x, self.samplingRate,
                                    windowType=self.spectrumWindowType,
                                    windowLength=windowLength)
        if method == "bartlett":
            return getBartlett(x, self.samplingRate, numSegment=self.nSpectrumSegments)
        if method == "welch":
            return getWelch(x, self.samplingRate,
                            numSegment=self.nSpectrumSegments,
                            windowType=self.spectrumWindowType,
                            overlap=self.spectrumOverlap)
        raise ValueError(f"Unsupported spectrum method")

    def _formatSpectrum(self, psd, frequencies):
        if self.spectrumFormat == "psd":
            spectrum = psd
        else:
            power = psd * (frequencies[1] - frequencies[0])
            if self.spectrumFormat == "power":
                spectrum = power
            else:
                spectrum = np.sqrt(np.maximum(power, 0))

        if self.spectrumDisplayMode == "linear":
            return spectrum

        multiplier = 20 if self.spectrumFormat == "linear" else 10
        return multiplier * np.log10(np.maximum(spectrum, np.finfo(float).eps))

    def debugSnapshotSpectrum(self, x):
        if x.size == 0:
            print("Empty snapshot")
            return
        if self.spectrumFormat not in ("psd", "power"): return

        psd, frequencies = self._estPSD(x, self.spectrumSnapshotWindowLength)
        variance = np.mean((x - np.mean(x))**2)

        if self.spectrumFormat == "psd":
            estimate = np.trapezoid(psd, frequencies)
        else:
            estimate = np.sum(psd * (frequencies[1] - frequencies[0]))

        error = estimate - variance
        error = error / variance if variance > 0 else 0
        print(f"variance = {variance:.3e}, error = {error:.2%}")

    def estSpectrogram(self, x, windowLength):
        spectrogram, times, frequencies = getSpectrogram(x, self.samplingRate,
                                                         windowLength=windowLength,
                                                         overlap=self.spectrogramOverlap,
                                                         windowType=self.spectrogramWindowType)
        return spectrogram[frequencies <= MAX_FREQUENCY], times, frequencies[frequencies <= MAX_FREQUENCY]

    def pause(self):
        with self.bufferLock:
            self.captureEnabled.clear()
            samples = self.getSamples(self.nSnapshotSamples, pad=False)
            wave = samples[-self.nWaveformSamples:]
            spectrumSamples = samples[-self.nSpectrumSamples:]
        return samples, wave, spectrumSamples
    
    def resume(self):
        self.captureEnabled.set()
        with self.snapshotLock:
            self.snapshotWave = None
            self.snapshotSpectrumSamples = None
            self.snapshotSpectrumFreq = None
            self.snapshotSpectrum = None
        
    def update(self, frame):
        if self.closingEvent.is_set() or not plt.fignum_exists(self.liveFig.number):
            return self.plotWave, self.plotSpectrum, self.plotSpectrogram, self.statusLine
        
        while True:
            try:
                print(f"Warning: {self._callbackStatus.popleft()}")
            except IndexError: break

        with self.snapshotLock:
            snapshotWave = self.snapshotWave
            snapshotSpectrumFreq = self.snapshotSpectrumFreq
            snapshotSpectrum = self.snapshotSpectrum

        if snapshotWave is None:
            with self.bufferLock:
                if self.nWaveformSamples >= self.nSpectrumSamples:
                    wave = self.getSamples(self.nWaveformSamples)
                    spectrumSamples = wave[-self.nSpectrumSamples:]
                else:
                    spectrumSamples = self.getSamples(self.nSpectrumSamples)
                    wave = spectrumSamples[-self.nWaveformSamples:]

            # if (self.frameIndex % self.spectrumRefreshRate) == 0:
            #     self.liveSpectrum, self.liveSpectrumFreq = self.estSpectrum(spectrumSamples, self.spectrumWindowLength)
            #     self.liveSpectrogram, self.liveSpectrogramTimes, self.liveSpectrogramFreq = self.estSpectrogram(spectrumSamples, windowLength=self.spectrogramWindowLength)
            if (self.frameIndex % self.spectrumRefreshRate) == 0:
                if SPECTRUM_ALPHA == 0:
                    self.liveSpectrum, self.liveSpectrumFreq = self.estSpectrum(spectrumSamples, self.spectrumWindowLength)
                else:
                    psd, self.liveSpectrumFreq = self._estPSD(spectrumSamples, self.spectrumWindowLength)
                    if self.livePSD.shape != psd.shape:
                        self.livePSD = psd
                    else:
                        self.livePSD = psd * SPECTRUM_ALPHA + (1 - SPECTRUM_ALPHA) * self.livePSD
                    self.liveSpectrum = self._formatSpectrum(self.livePSD, self.liveSpectrumFreq)
            if (self.frameIndex % self.spectrogramRefreshRate) == 0:
                self.liveSpectrogram, _, _ = self.estSpectrogram(spectrumSamples, windowLength=self.spectrogramWindowLength)

            self.frameIndex += 1
            spectrumFreq = self.liveSpectrumFreq
            spectrum = self.liveSpectrum

        else:
            wave = snapshotWave
            spectrumFreq = snapshotSpectrumFreq
            spectrum = snapshotSpectrum
        
        if WAVEFORM_PLOT_POINTS == 0:
            pass
        elif WAVEFORM_ENVELOPE:
            wave = self._waveformEnvelope(wave)
        else:
            wave = wave[::self.waveStride][:len(self.waveTime)]
        self.plotWave.set_data(self.waveTime[:len(wave)], wave)
        self.plotSpectrum.set_xdata(spectrumFreq)
        self.plotSpectrum.set_ydata(spectrum)
        self.plotSpectrogram.set_data(self.liveSpectrogram)
        # self.plotSpectrogram.autoscale()

        self.statusLine.set_text(f"{'LIVE' if snapshotWave is None else 'HOLD'} | {int(self.samplingRate)} Hz | {self.blockSize} | {self.device}/{self.outputDevice}")

        return self.plotWave, self.plotSpectrum, self.plotSpectrogram, self.statusLine

    def stop(self):
        self._shutdown()
        if plt.fignum_exists(self.liveFig.number):
            plt.close(self.liveFig)

    def onKey(self, event):
        if event.key == "s":
            raw, wave, spectrumSamples = self.pause()
            if raw.size == 0:
                self.captureEnabled.set()
                print("No audio captured")
                return

            if SPECTRUM_DEBUG:
                self.debugSnapshotSpectrum(raw)

            spectrum, spectrumFreq = self.estSpectrum(spectrumSamples, self.spectrumSnapshotWindowLength)

            with self.snapshotLock:
                self.snapshotWave = wave
                self.snapshotSpectrumSamples = spectrumSamples
                self.snapshotSpectrumFreq = spectrumFreq
                self.snapshotSpectrum = spectrum

            try:
                processed = self.processAudio(raw)
            except ValueError as err:
                self.captureEnabled.set()
                print(err)
                return

            with self.snapshotLock:
                self.snapshotSamples = raw
                self.snapshotProcessed = processed
            print(f"Saved snapshot")

        elif event.key == "l":
            self.resume()
            print("Resumed")

        elif event.key == "r":
            with self.snapshotLock:
                raw = self.snapshotSamples
            self.queueSnapshot(raw, "raw snapshot")

        elif event.key == "p":
            with self.snapshotLock:
                processed = self.snapshotProcessed
            self.queueSnapshot(processed, "processed snapshot")
        
        elif event.key == "c":
            self.plotSnapshot()

        elif event.key == "w":
            self.saveSnapshot()

        elif event.key == "q":
            self.stop()

        elif event.key == "d":
            self.toggleSpectrumDisplayMode()

        elif event.key == "n":
            self.cycleSpectrumFormat()
        elif event.key == "N":
            self.cycleSpectrumFormat(-1)
    
    def _cleanup(self):
        with self.inputStreamLock:
            try:
                if self.inputStream.active: self.inputStream.stop()
            except Exception as err:
                print(err)
            try:
                self.inputStream.close()
            except Exception as err:
                print(err)
        
        with self.outputStreamLock:
            try:
                if self.outputStream.active: self.outputStream.stop()
            except Exception as err:
                print(err)
            try:
                self.outputStream.close()
            except Exception as err:
                print(err)
    
    def _shutdown(self, event=None):
        if self.closingEvent.is_set():
            return
        
        self.closingEvent.set()
        if self.ani is not None and self.ani.event_source is not None:
            self.ani.event_source.stop()

        print("Stream stopped")
        self._cleanup()

    def saveSnapshot(self):
        with self.snapshotLock:
            raw = self.snapshotSamples
            processed = self.snapshotProcessed

        if raw is None or processed is None:
            print("No snapshot available")
            return
        
        sp.io.wavfile.write("raw.wav", self.samplingRate, self.floatToInt16(raw))
        sp.io.wavfile.write("processed.wav", self.samplingRate, self.floatToInt16(processed))
        print("Snapshot saved")

    def queueSnapshot(self, x, snapshotType):
        if x is None:
            print("No snapshot available")
            return

        with self.playbackLock:
            self._pendingPlayback = (x.copy(), 0)
        print(f"Queued {snapshotType}")

    def plotSnapshot(self):
        with self.snapshotLock:
            raw = self.snapshotSamples
            processed = self.snapshotProcessed

        if raw is None or processed is None:
            print("No snapshot available")
            return

        t = np.arange(len(raw)) / self.samplingRate

        rawSpectrum, spectrumFreq = self.estSpectrum(raw, self.spectrumSnapshotWindowLength)
        processedSpectrum, _ = self.estSpectrum(processed, self.spectrumSnapshotWindowLength)

        rawSpectrogram, specTimes, specFreq = self.estSpectrogram(raw, self.spectrogramSnapshotWindowLength)
        processedSpectrogram, _, _ = self.estSpectrogram(processed, self.spectrogramSnapshotWindowLength)

        if self.snapshotFig is None or not plt.fignum_exists(self.snapshotFig.number):
            self.snapshotFig, self.snapshotAxs = plt.subplots(2, 2, figsize=(12, 5))
            axWave, axSpectrum = self.snapshotAxs[:, 0]
            axSpectrogramRaw, axSpectrogramProcessed = self.snapshotAxs[:, 1]

            self.snapshotWaveRaw, = axWave.plot([], [])
            self.snapshotWaveProcessed, = axWave.plot([], [])
            axWave.set_title("Snapshot waveform")
            axWave.set_xlabel("Time (s)")
            axWave.set_ylabel("Amplitude")
            axWave.set_xlim(0, self.snapshotDuration)
            axWave.set_ylim(-1, 1)

            self.snapshotSpectrumRaw, = axSpectrum.plot([], [])
            self.snapshotSpectrumProcessed, = axSpectrum.plot([], [])
            axSpectrum.set_title(f"Snapshot spectrum - {self.spectrumMethod.upper()} / {self.spectrumWindowType.upper()}")
            axSpectrum.set_xlabel("Frequency (Hz)")
            axSpectrum.set_xlim(20, MAX_FREQUENCY)
            self._applySpectrumStyle(axSpectrum)

            self.snapshotSpectrogramRaw = axSpectrogramRaw.imshow(rawSpectrogram, origin="lower", aspect="auto", cmap="turbo",
                                                                  extent=(specTimes[0], specTimes[-1], specFreq[0], specFreq[-1]),
                                                                  interpolation="nearest")
            axSpectrogramRaw.set_title("Raw spectrogram")
            axSpectrogramRaw.set_xlabel("Time (s)")
            axSpectrogramRaw.set_ylabel("Frequency (Hz)")
            self.snapshotSpectrogramProcessed = axSpectrogramProcessed.imshow(processedSpectrogram, origin="lower", aspect="auto", cmap="turbo",
                                                                              extent=(specTimes[0], specTimes[-1], specFreq[0], specFreq[-1]),
                                                                              interpolation="nearest")
            axSpectrogramProcessed.set_title("Processed spectrogram")
            axSpectrogramProcessed.set_xlabel("Time (s)")
            axSpectrogramProcessed.set_ylabel("Frequency (Hz)")

            self.snapshotFig.tight_layout()

        self._applySpectrumStyle(self.snapshotAxs[1, 0])

        self.snapshotWaveRaw.set_data(t, raw)
        self.snapshotWaveProcessed.set_data(t, processed)

        self.snapshotSpectrumRaw.set_data(spectrumFreq, rawSpectrum)
        self.snapshotSpectrumProcessed.set_data(spectrumFreq, processedSpectrum)

        self.snapshotSpectrogramRaw.set_data(rawSpectrogram)
        self.snapshotSpectrogramProcessed.set_data(processedSpectrogram)

        self.snapshotFig.canvas.draw()
        manager = getattr(self.snapshotFig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "show"):
            manager.show()

    def toggleSpectrumDisplayMode(self):
        if self.spectrumDisplayMode == "linear":
            self.spectrumDisplayMode = "db"
        else:
            self.spectrumDisplayMode = "linear"
        self._refreshSpectrumDisplay()

    def cycleSpectrumFormat(self, direction=1):
        i = SPECTRUM_FORMATS.index(self.spectrumFormat)
        self.spectrumFormat = SPECTRUM_FORMATS[(i + direction) % len(SPECTRUM_FORMATS)]
        self._refreshSpectrumDisplay()

    def _spectrumAxisLabel(self):
        if self.spectrumFormat == "linear":
            label = "RMS amplitude"
        elif self.spectrumFormat == "power":
            label = "Power"
        elif self.spectrumFormat == "psd":
            label = "PSD"
        return f"{label} (dB)" if self.spectrumDisplayMode == "db" else label

    def _spectrumAxisLimits(self):
        if self.spectrumDisplayMode == "db":
            return SPECTRUM_DB_LIM
        return SPECTRUM_LINEAR_LIM[self.spectrumFormat]

    def _applySpectrumStyle(self, ax):
        ax.set_ylabel(self._spectrumAxisLabel())
        ax.set_ylim(self._spectrumAxisLimits())
        ax.set_xscale("log" if self.spectrumDisplayMode == "db" else "linear")

    def _refreshSnapshotSpectrum(self):
        with self.snapshotLock:
            raw = self.snapshotSamples
            processed = self.snapshotProcessed

        if raw is None or processed is None: return

        rawSpectrum, spectrumFreq = self.estSpectrum(raw, self.spectrumSnapshotWindowLength)
        processedSpectrum, _ = self.estSpectrum(processed, self.spectrumSnapshotWindowLength)
        self.snapshotSpectrumRaw.set_data(spectrumFreq, rawSpectrum)
        self.snapshotSpectrumProcessed.set_data(spectrumFreq, processedSpectrum)

    def _refreshSpectrumDisplay(self):
        self.plotSpectrum.set_data([], [])
        if self.snapshotSpectrumRaw is not None:
            self.snapshotSpectrumRaw.set_data([], [])
        if self.snapshotSpectrumProcessed is not None:
            self.snapshotSpectrumProcessed.set_data([], [])

        # Live
        self._applySpectrumStyle(self.axSpectrum)
        self.liveFig.canvas.draw_idle()

        if self.snapshotFig and plt.fignum_exists(self.snapshotFig.number):
            self._applySpectrumStyle(self.snapshotAxs[1, 0])
            self._refreshSnapshotSpectrum()
            self.snapshotFig.canvas.draw_idle()

        with self.bufferLock:
            spectrumSamples = self.getSamples(self.nSpectrumSamples)
        psd, self.liveSpectrumFreq = self._estPSD(spectrumSamples, self.spectrumWindowLength)
        self.livePSD = psd
        self.liveSpectrum = self._formatSpectrum(self.livePSD, self.liveSpectrumFreq)

        # Snapshot
        with self.snapshotLock:
            samples = self.snapshotSpectrumSamples
        if samples is None:
            self.frameIndex = 0
            return

        spectrum, spectrumFreq = self.estSpectrum(samples, self.spectrumWindowLength)
        with self.snapshotLock:
            self.snapshotSpectrum = spectrum
            self.snapshotSpectrumFreq = spectrumFreq

        self.frameIndex = 0

    def _waveformEnvelope(self, wave):
        if WAVEFORM_PLOT_POINTS == 0:
            return wave

        wave = wave[:len(wave) // self.waveStride * self.waveStride]
        samples = wave.reshape(len(wave) // self.waveStride, self.waveStride)
        envelopeMin, envelopeMax = samples.min(axis=1), samples.max(axis=1)

        envelope = np.empty(2 * len(wave) // self.waveStride, dtype=wave.dtype)
        envelope[0::2] = envelopeMin
        envelope[1::2] = envelopeMax
        return envelope

    def processAudio(self, x):
        # x = x - np.mean(x)
        return sp.signal.sosfiltfilt(self.processFilter, x)
    
    @staticmethod
    def floatToInt16(x, normalize=False):
        if normalize and x.size:
            x = x / np.max(np.abs(x))
        else:
            x = np.clip(x, -1.0, 1.0)
        return (x * 32767).astype(np.int16)


    def run(self):
        for k in CONFLICT_KEYMAPS:
            plt.rcParams[k] = []

        self.liveFig.canvas.mpl_connect("key_press_event", self.onKey)
        self.liveFig.canvas.mpl_connect("close_event", self._shutdown)

        self.ani = FuncAnimation(self.liveFig, self.update, interval=20, blit=False, cache_frame_data=False)

        print("Stream started")

        with self.outputStreamLock:
            self.outputStream.start()
        with self.inputStreamLock:
            self.inputStream.start()

        try:
            plt.tight_layout()
            plt.show()
        finally:
            self._shutdown()

        # with self.snapshotLock:
        #     hasSnapshot = self.snapshotSamples and self.snapshotProcessed 


if __name__ == "__main__":
    stream = realtimeTracking()
    stream.run()
