from typing import Optional
import threading
import collections
import time

import numpy as np
import scipy as sp
import sounddevice as sd

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from spectralAnalysis import getACF, getWindow


RATE = 44100
DEVICE = 1
OUTPUT_DEVICE = 0
CHANNELS = 1
BLOCK_SIZE = 1024

WAVEFORM_ENVELOPE = True
WAVEFORM_PLOT_POINTS = 128

WAVEFORM_DURATION = 0.5
BUFFER_DURATION = 3.0
SNAPSHOT_DURATION = 3.0

SPECTRUM_WINDOW = "rect"
NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

PITCH_REFRESH_DURATION = 0.5
PITCH_THRESHOLD_DB = -40
PITCH_WINDOW_DURATION = 0.15
PITCH_HOLD_DURATION = 3.0

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
                 bufferDuration: Optional[float] = BUFFER_DURATION,
                 snapshotDuration: Optional[float] = SNAPSHOT_DURATION,
                 pitchWindowDuration: Optional[float] = PITCH_WINDOW_DURATION,
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
        self.bufferDuration = bufferDuration
        self.snapshotDuration = snapshotDuration
        self.pitchWindowDuration = pitchWindowDuration

        self.nWaveformSamples = int(samplingRate * self.waveformDuration)
        self.nBufferSamples = int(samplingRate * self.bufferDuration)
        self.nSnapshotSamples = int(samplingRate * self.snapshotDuration)
        self.nPitchSamples = int(samplingRate * self.pitchWindowDuration)

        if self.nSnapshotSamples > self.nBufferSamples:
            raise ValueError("Snapshot duration cannot exceed buffer length.")

        self.buffer = np.zeros(self.nBufferSamples, dtype=np.float32)
        self.writePos = 0
        self.filledSamples = 0

        self.inputStreamLock = threading.Lock()
        self.outputStreamLock = threading.Lock()
        self.bufferLock = threading.Lock()
        self.snapshotLock = threading.Lock()
        self.playbackLock = threading.Lock()

        self._callbackStatus = collections.deque(maxlen=32)
        self._pendingPlayback = None
        self._activePlayback = None

        self.lastPitch = None
        self.lastPitchUpdate = 0
        self.lastPitchValidTime = 0

        self.captureEnabled = threading.Event()
        self.captureEnabled.set()
        self.closingEvent = threading.Event()

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

        self.liveFig, (self.axWave, self.axStatus) = plt.subplots(2, 1, figsize=(5, 5), gridspec_kw={"height_ratios": [4, 1]})
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

        self.plotWave, = self.axWave.plot(self.waveTime, np.zeros_like(self.waveTime))
        self.axWave.set_title("Live waveform")
        self.axWave.set_xlabel("Time (s)")
        self.axWave.set_ylabel("Amplitude")
        self.axWave.set_xlim(0, self.waveformDuration)
        self.axWave.set_ylim(-1, 1)
        self.axWave.grid(True)

        self.axStatus.set_axis_off()
        self.statusLine = self.axStatus.text(0.42, 0.5, "Listening...", transform=self.axStatus.transAxes,
                                             ha="center", va="bottom", fontsize=16, fontweight="bold")

        self.snapshotFig, self.snapshotAx = None, None
        self.snapshotWave = None

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

    def estPitch(self, x):
        if x.size == 0:
            return None
        
        x = x - np.mean(x)
        power = np.mean(x**2)
        if 10 * np.log10(np.maximum(power, np.finfo(float).eps)) < PITCH_THRESHOLD_DB:
            return None

        minFreq = 20.0
        maxFreq = 1200.0

        minLag = max(1, int(self.samplingRate / maxFreq))
        maxLag = min(len(x) - 1, int(self.samplingRate / minFreq))
        
        r = getACF(x, maxLag)
        searchRange = r[minLag:maxLag + 1]
        peak = int(np.argmax(searchRange))

        delta = 0.5 * (searchRange[peak - 1] - searchRange[peak + 1]) / (searchRange[peak - 1] - 2 * searchRange[peak] + searchRange[peak + 1])
        peak += delta + minLag

        f0 = self.samplingRate / peak
        midi = 69 + 12.0 * np.log2(f0 / 440.0)
        c = (midi - int(midi)) * 100
        if c > 50:
            c -= 100
            midi += 1

        return f"{NOTES[int(midi) % 12]}{int(midi) // 12 - 1} {c:+.1f}"

    def pause(self):
        with self.bufferLock:
            self.captureEnabled.clear()
            samples = self.getSamples(self.nSnapshotSamples, pad=False)
            wave = samples[-self.nWaveformSamples:]

        self.estPitch(wave)
        return samples, wave

    def resume(self):
        self.captureEnabled.set()
        with self.snapshotLock:
            self.snapshotWave = None

    def update(self, frame):
        if self.closingEvent.is_set() or not plt.fignum_exists(self.liveFig.number):
            return self.plotWave, self.statusLine
        
        while True:
            try:
                print(f"Warning: {self._callbackStatus.popleft()}")
            except IndexError: break

        with self.snapshotLock:
            snapshotWave = self.snapshotWave

        if snapshotWave is None:
            with self.bufferLock:
                wave = self.getSamples(self.nWaveformSamples)
                wavePitch = self.getSamples(self.nPitchSamples)

        else:
            wave = snapshotWave[-self.nWaveformSamples:]
            wavePitch = snapshotWave[-self.nPitchSamples:]

        if WAVEFORM_PLOT_POINTS != 0:
            if WAVEFORM_ENVELOPE:
                wave = self._waveformEnvelope(wave)
            else:
                wave = wave[::self.waveStride][:len(self.waveTime)]

        self.plotWave.set_data(self.waveTime[:len(wave)], wave)

        now = time.monotonic()
        if (now - self.lastPitchUpdate) >= PITCH_REFRESH_DURATION:
            self.lastPitchUpdate = now
            result = self.estPitch(wavePitch)
            if result:
                self.lastPitch = result
                self.lastPitchValidTime = now

        if (now - self.lastPitchValidTime) < PITCH_HOLD_DURATION:
            self.statusLine.set_text(self.lastPitch)
        else:
            self.statusLine.set_text("Listening...")

        return self.plotWave, self.statusLine

    def stop(self):
        self._shutdown()
        if plt.fignum_exists(self.liveFig.number):
            plt.close(self.liveFig)

    def onKey(self, event):
        if event.key == "s":
            raw, wave = self.pause()
            if raw.size == 0:
                self.capturedEnabled.set()
                print("No audio captured")
                return
            
            with self.snapshotLock:
                self.snapshotWave = raw
            print(f"Saved snapshot")

        elif event.key == "l":
            self.resume()
            print("Resumed")

        elif event.key == "r":
            with self.snapshotLock:
                raw = self.snapshotWave
            self.queueSnapshot(raw, "snapshot")

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

    def queueSnapshot(self, x, snapshotType):
        if x is None:
            print("No snapshot available")
            return

        with self.playbackLock:
            self._pendingPlayback = (x.copy(), 0)
        print(f"Queued {snapshotType}")

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


if __name__ == "__main__":
    stream = realtimeTracking()
    stream.run()
