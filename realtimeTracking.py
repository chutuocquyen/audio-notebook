from typing import Optional
import threading
import collections
import queue

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation

from spectralAnalysis import *


RATE = 44100
DEVICE = 1
OUTPUT_DEVICE = 2
CHANNELS = 1
BLOCK_SIZE = 1024
WINDOW_TYPE = "hann"
PSD_METHOD = "bartlett"

WAVEFORM_DURATION = 0.5
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
                 samplingRate: Optional[float] = RATE,
                 waveformDuration: Optional[float] = WAVEFORM_DURATION,
                 spectrumDuration: Optional[float] = SPECTRUM_DURATION,
                 bufferDuration: Optional[float] = BUFFER_DURATION,
                 snapshotDuration: Optional[float] = SNAPSHOT_DURATION,
                 windowType: Optional[str] = WINDOW_TYPE,
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

        self._callbackStatus = collections.deque(maxlen=32)
        self._pendingPlayback = queue.SimpleQueue()
        self._activePlayback = None
        self._silence = np.zeros(BLOCK_SIZE)

        self.captureEnabled = True
        self.closingEvent = threading.Event()

        self.snapshotRaw = None
        self.snapshotProcessed = None
        self.frozenWave = None
        self.frozenSpectrum = None

        self.windowType = windowType
        self.spectrumWindow = getWindow(windowType, self.nSpectrumSamples)

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

        self.fig, (self.axWave, self.axSpec) = plt.subplots(2, 1, figsize=(15, 5))
        self.ani = None

        waveTime = np.arange(self.nWaveformSamples) / self.samplingRate
        specFreq = np.fft.rfftfreq(self.nSpectrumSamples, d=1.0 / self.samplingRate)

        self.plotWave, = self.axWave.plot(waveTime, np.zeros(self.nWaveformSamples))
        self.axWave.set_xlabel("Time (s)")
        self.axWave.set_ylabel("Amplitude")
        self.axWave.set_xlim(0, self.waveformDuration)
        self.axWave.set_ylim(-1, 1)

        self.plotSpec, = self.axSpec.plot(specFreq, np.zeros_like(specFreq))
        self.axSpec.set_xlabel("Frequency (Hz)")
        self.axSpec.set_ylabel("Magnitude")
        self.axSpec.set_xlim(20, 5000)
        self.axSpec.set_ylim(-1e-6, 1.2e-4)
        # self.axSpec.set_ylim(-120, 10)

    def audioCallback(self, indata, frames, time, status):
        if status:
            self._callbackStatus.append(str(status))

        if not self.captureEnabled: return

        x = indata[:, 0]
        signalLength = len(x)
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

        try:
            self._activePlayback = self._pendingPlayback.get_nowait()
        except queue.Empty:
            pass

        outdata[:, 0] = self._silence[:frames]
        if self._activePlayback is not None:
            buf, pos = self._activePlayback
            count = min(frames, len(buf) - pos)
            if count > 0:
                outdata[:count, 0] = buf[pos:pos + count]
                pos += count
            self._activePlayback = (buf, pos) if pos < len(buf) else None

    def getSamples(self, nSamples, pad=True):
        available = min(nSamples, self.filledSamples)
        if available == 0:
            if pad:
                return np.zeros(nSamples)
            return np.zeros(0)

        i0 = (self.writePos - available) % self.nBufferSamples
        endPos = i0 + available
        if endPos <= self.nBufferSamples:
            samples = self.buffer[i0:endPos].copy()
        else:
            tmp = self.nBufferSamples - i0
            samples = np.concatenate((self.buffer[i0:], self.buffer[:available - tmp])).copy()

        if not pad or available == nSamples:
            return samples
        padded = np.zeros(nSamples)
        padded[-available:] = samples
        return padded

    def estPSD(self, x,
               method: Optional[str] = PSD_METHOD,
               nSegment: Optional[int] = 10,
               windowType: Optional[str] = "hann",
               overlap: Optional[float] = 0.5,
               windowLength: Optional[int] = 10001):
        x = x - np.mean(x)
        method = method.strip().lower()

        if method == "correlogram":
            psd, frequencies = getCorrelogram(x, self.samplingRate)

        elif method == "periodogram":
            psd, frequencies = getPeriodogram(x, self.samplingRate)
        elif method == "blackmantukey":
            psd, frequencies = getBlackmanTukey(x, self.samplingRate,
                                                windowType=windowType,
                                                windowLength=windowLength)
        elif method == "bartlett":
            psd, frequencies = getBartlett(x, self.samplingRate, numSegment=nSegment)
        elif method == "welch":
            psd, frequencies = getWelch(x, self.samplingRate,
                                        numSegment=nSegment,
                                        windowType=windowType,
                                        overlap=overlap)
        else:
            raise ValueError(f"Unsupported PSD method")

        targetFrequencies = np.fft.rfftfreq(len(x), d=1.0 / self.samplingRate)
        psd = np.interp(targetFrequencies, frequencies, psd)
        # return 10 * np.log10(np.maximum(psd, 1e-12))
        return psd

    def pause(self):
        self.captureEnabled = False
        raw = self.getSamples(self.nSnapshotSamples, pad=False)
        wave = self.getSamples(self.nWaveformSamples, pad=True)
        spec = self.getSamples(self.nSpectrumSamples, pad=True)
        return raw, wave, spec
    
    def resume(self):
        self.captureEnabled = True
        with self.snapshotLock:
            self.frozenWave = None
            self.frozenSpectrum = None
        
    def update(self, frame):
        if self.closingEvent.is_set() or not plt.fignum_exists(self.fig.number):
            return self.plotWave, self.plotSpec
        
        while self._callbackStatus:
            print(f"Warning: {self._callbackStatus.popleft()}")

        with self.snapshotLock:
            frozenWave = None if self.frozenWave is None else self.frozenWave.copy()
            frozenSpectrum = None if self.frozenSpectrum is None else self.frozenSpectrum.copy()

        if frozenWave is None:
            wave = self.getSamples(self.nWaveformSamples)
            spec = self.getSamples(self.nSpectrumSamples)
        else:
            wave = frozenWave
            spec = frozenSpectrum

        spec = self.estPSD(spec, method=PSD_METHOD, windowType=self.windowType)
        
        self.plotWave.set_ydata(wave)
        self.plotSpec.set_ydata(spec)

        return self.plotWave, self.plotSpec

    def onKey(self, event):
        if event.key == "s":
            raw, wave, spec = self.pause()
            if raw.size == 0:
                self.captureEnabled = True
                print("No audio captured")
                return

            with self.snapshotLock:
                self.frozenWave = wave
                self.frozenSpectrum = spec

            try:
                processed = self.processAudio(raw, self.samplingRate)
            except ValueError as err:
                self.captureEnabled = True
                with self.snapshotLock:
                    self.frozenWave = wave
                    self.frozenSpectrum = spec
                print(err)
                return

            with self.snapshotLock:
                self.snapshotRaw = raw
                self.snapshotProcessed = processed
            print(f"Saved snapshot")

        elif event.key == "l":
            self.resume()
            print("Resumed")

        elif event.key == "r":
            with self.snapshotLock:
                raw = None if self.snapshotRaw is None else self.snapshotRaw.copy()
            self.queueSnapshot(raw, "raw snapshot")

        elif event.key == "p":
            with self.snapshotLock:
                processed = None if self.snapshotProcessed is None else self.snapshotProcessed.copy()
            self.queueSnapshot(processed, "processed snapshot")
        
        elif event.key == "c":
            self.plotSnapshot()

        elif event.key == "w":
            self.saveSnapshot()

        elif event.key == "q":
            self.stop()
    
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

    def stop(self):
        self._shutdown()
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

    def saveSnapshot(self):
        with self.snapshotLock:
            raw = None if self.snapshotRaw is None else self.snapshotRaw.copy()
            processed = None if self.snapshotProcessed is None else self.snapshotProcessed.copy()

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
        self._pendingPlayback.put((x.copy(), 0))
        print(f"Queued {snapshotType}")

    def plotSnapshot(self):
        with self.snapshotLock:
            raw = None if self.snapshotRaw is None else self.snapshotRaw.copy()
            processed = None if self.snapshotProcessed is None else self.snapshotProcessed.copy()

        if raw is None or processed is None:
            print("No snapshot available")
            return
        
        t = np.arange(len(raw)) / self.samplingRate

        rawPSD = self.estPSD(raw)
        processedPSD = self.estPSD(processed)
        f = np.fft.rfftfreq(len(raw), d=1.0 / self.samplingRate)

        _, axs = plt.subplots(2, 1, figsize=(20, 10))
        axs = axs.flatten()

        axs[0].plot(t, raw)
        axs[0].plot(t, processed)
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_xlim(0, len(raw) / self.samplingRate)
        
        axs[1].plot(f, rawPSD)
        axs[1].plot(f, processedPSD)
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Magnitude")
        axs[1].set_xlim(0, 5000)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def processAudio(x, samplingRate):
        x = x - np.mean(x)
        filter = sp.signal.butter(2, [100, 2500], btype="bandpass", fs=samplingRate, output="sos")

        y = sp.signal.sosfiltfilt(filter, x)
        return y
    
    @staticmethod
    def floatToInt16(x):
        x = np.clip(x, -1.0, 1.0)
        return (x * 32767).astype(np.int16)

    def run(self):
        for k in CONFLICT_KEYMAPS:
            plt.rcParams[k] = []

        self.fig.canvas.mpl_connect("key_press_event", self.onKey)
        self.fig.canvas.mpl_connect("close_event", self._shutdown)

        self.ani = FuncAnimation(self.fig, self.update, interval=50, blit=False, cache_frame_data=False)

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
        #     hasSnapshot = self.snapshotRaw and self.snapshotProcessed 


if __name__ == "__main__":
    stream = realtimeTracking()
    stream.run()
