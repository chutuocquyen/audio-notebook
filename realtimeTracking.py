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
OUTPUT_DEVICE = 0
CHANNELS = 1
BLOCK_SIZE = 1024
WINDOW_TYPE = "hann"
PSD_METHOD = "bartlett"

WAVEFORM_DURATION = 0.5
PSD_DURATION = 0.5
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
                 psdDuration: Optional[float] = PSD_DURATION,
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
        self.psdDuration = psdDuration
        self.bufferDuration = bufferDuration
        self.snapshotDuration = snapshotDuration

        self.nWaveformSamples = int(samplingRate * self.waveformDuration)
        self.nPSDSamples = int(samplingRate * self.psdDuration)
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
        self._silence = np.zeros(self.blockSize, dtype=np.float32)

        self.captureEnabled = True
        self.closingEvent = threading.Event()

        self.snapshotRaw = None
        self.snapshotProcessed = None

        self.frozenWave = None
        self.frozenPSDSamples = None
        self.frozenPSD = None

        self.windowType = windowType

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

        self.fig, axs = plt.subplots(2, 2, figsize=(8, 5))
        self.renderer = self.fig.canvas.get_renderer()
        self.axWave, self.axPSD = axs[0]
        self.axSnapshotWave, self.axSnapshotPSD = axs[1]
        self.ani = None

        self.waveTime = np.arange(self.nWaveformSamples) / self.samplingRate
        self.psdFreq = np.fft.rfftfreq(self.nPSDSamples, d=1.0 / self.samplingRate)
        self.snapshotPSDFreq = np.fft.rfftfreq(self.nSnapshotSamples, d=1.0 / self.samplingRate)

        self.frameIndex = 0
        self.refreshRate = 3
        self.livePSD = np.zeros(len(self.psdFreq))

        self.plotWave, = self.axWave.plot(self.waveTime, np.zeros(self.nWaveformSamples))
        self.axWave.set_title("Live waveform")
        self.axWave.set_xlabel("Time (s)")
        self.axWave.set_ylabel("Amplitude")
        self.axWave.set_xlim(0, self.waveformDuration)
        self.axWave.set_ylim(-1, 1)

        self.plotPSD, = self.axPSD.plot(self.psdFreq, np.zeros_like(self.psdFreq))
        self.axPSD.set_title("Live PSD")
        self.axPSD.set_xlabel("Frequency (Hz)")
        self.axPSD.set_ylabel("Magnitude")
        self.axPSD.set_xlim(20, 5000)
        self.axPSD.set_ylim(-1e-6, 1.2e-5)
        # self.axPSD.set_ylim(-120, 10)

        self.snapshotWaveRaw, = self.axSnapshotWave.plot([], [])
        self.snapshotWaveProcessed, = self.axSnapshotWave.plot([], [])
        self.axSnapshotWave.set_title("Snapshot waveform")
        self.axSnapshotWave.set_xlabel("Time (s)")
        self.axSnapshotWave.set_ylabel("Amplitude")
        self.axSnapshotWave.set_xlim(0, self.snapshotDuration)
        self.axSnapshotWave.set_ylim(-1, 1)

        self.snapshotPSDRaw, = self.axSnapshotPSD.plot([], [])
        self.snapshotPSDProcessed, = self.axSnapshotPSD.plot([], [])
        self.axSnapshotPSD.set_title("Snapshot PSD")
        self.axSnapshotPSD.set_xlabel("Frequency (Hz)")
        self.axSnapshotPSD.set_ylabel("Magnitude")
        self.axSnapshotPSD.set_xlim(20, 5000)
        self.axSnapshotPSD.set_ylim(-1e-6, 1.2e-5)

    def audioCallback(self, indata, frames, time, status):
        if status:
            self._callbackStatus.append(str(status))

        if not self.captureEnabled: return

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
                return np.zeros(nSamples, dtype=np.float32)
            return np.zeros(0, dtype=np.float32)

        i0 = (self.writePos - available) % self.nBufferSamples
        endPos = i0 + available

        if endPos <= self.nBufferSamples:
            samples = self.buffer[i0:endPos].copy()
        else:
            tmp = self.nBufferSamples - i0
            samples = np.concatenate((self.buffer[i0:], self.buffer[:available - tmp])).copy()

        if not pad or available == nSamples:
            return samples
        padded = np.zeros(nSamples, dtype=np.float32)
        padded[-available:] = samples
        return padded

    def estPSD(self, x,
               method: Optional[str] = PSD_METHOD,
               nSegment: Optional[int] = 10,
               windowType: Optional[str] = "hann",
               overlap: Optional[float] = 0.5,
               windowLength: Optional[int] = 10001,
               targetFrequencies=None):
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

        if targetFrequencies is None:
            return psd
        if np.array_equal(frequencies, targetFrequencies):
            return psd

        psd = np.interp(targetFrequencies, frequencies, psd)
        # return 10 * np.log10(np.maximum(psd, 1e-12))
        return psd

    def pause(self):
        self.captureEnabled = False
        with self.bufferLock:
            raw = self.getSamples(self.nSnapshotSamples, pad=False)
            wave = self.getSamples(self.nWaveformSamples, pad=True)
            psdSamples = self.getSamples(self.nPSDSamples, pad=True)
        return raw, wave, psdSamples
    
    def resume(self):
        self.captureEnabled = True
        with self.snapshotLock:
            self.frozenWave = None
            self.frozenPSDSamples = None
            self.frozenPSD = None
        
    def update(self, frame):
        if self.closingEvent.is_set() or not plt.fignum_exists(self.fig.number):
            return self.plotWave, self.plotPSD
        
        while self._callbackStatus:
            print(f"Warning: {self._callbackStatus.popleft()}")

        with self.snapshotLock:
            frozenWave = None if self.frozenWave is None else self.frozenWave.copy()
            frozenPSD = None if self.frozenPSD is None else self.frozenPSD.copy()

        if frozenWave is None:
            with self.bufferLock:
                wave = self.getSamples(self.nWaveformSamples)
                psdSamples = self.getSamples(self.nPSDSamples)

            if (self.frameIndex % self.refreshRate) == 0:
                self.livePSD = self.estPSD(psdSamples, method=PSD_METHOD, windowType=self.windowType, targetFrequencies=self.psdFreq)
            self.frameIndex += 1
            psd = self.livePSD

        else:
            wave = frozenWave
            psd = frozenPSD
        
        self.plotWave.set_ydata(wave)
        self.plotPSD.set_ydata(psd)

        return self.plotWave, self.plotPSD

    def stop(self):
        self._shutdown()
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

    def onKey(self, event):
        if event.key == "s":
            raw, wave, psdSamples = self.pause()
            if raw.size == 0:
                self.captureEnabled = True
                print("No audio captured")
                return

            psd = self.estPSD(psdSamples, method=PSD_METHOD, windowType=self.windowType, targetFrequencies=self.psdFreq)

            with self.snapshotLock:
                self.frozenWave = wave
                self.frozenPSDSamples = psdSamples
                self.frozenPSD = psd

            try:
                processed = self.processAudio(raw, self.samplingRate)
            except ValueError as err:
                self.captureEnabled = True
                with self.snapshotLock:
                    self.frozenWave = wave
                    self.frozenPSDSamples = psdSamples
                    self.frozenPSD = psd
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

        with self.playbackLock:
            self._pendingPlayback = (x.copy(), 0)
        print(f"Queued {snapshotType}")

    def plotSnapshot(self):
        with self.snapshotLock:
            raw = None if self.snapshotRaw is None else self.snapshotRaw.copy()
            processed = None if self.snapshotProcessed is None else self.snapshotProcessed.copy()

        if raw is None or processed is None:
            print("No snapshot available")
            return

        t = np.arange(len(raw)) / self.samplingRate
        f = self.snapshotPSDFreq

        rawPSD = self.estPSD(raw, targetFrequencies=self.snapshotPSDFreq)
        processedPSD = self.estPSD(processed, targetFrequencies=self.snapshotPSDFreq)

        self.snapshotWaveRaw.set_data(t, raw)
        self.snapshotWaveProcessed.set_data(t, processed)
        self.axSnapshotWave.set_xlim(0, len(raw) / self.samplingRate)

        self.snapshotPSDRaw.set_data(f, rawPSD)
        self.snapshotPSDProcessed.set_data(f, processedPSD)
        self.axSnapshotPSD.set_xlim(20, 5000)

        for i in [self.snapshotWaveRaw, self.snapshotWaveProcessed, self.snapshotPSDRaw, self.snapshotPSDProcessed]:
            i.draw(self.renderer)

    @staticmethod
    def processAudio(x, samplingRate):
        x = x - np.mean(x)
        filter = sp.signal.butter(2, [100, 2500], btype="bandpass", fs=samplingRate, output="sos")

        y = sp.signal.sosfiltfilt(filter, x)
        return y
    
    @staticmethod
    def floatToInt16(x):
        x = np.clip(x, -2.0, 2.0)
        return (x * 16383).astype(np.int16)

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
