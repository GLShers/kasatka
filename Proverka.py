import torchaudio

file_path = "дрон10_segment_77.wav"
waveform, sample_rate = torchaudio.load(file_path)
print(f"Sample rate: {sample_rate}, Channels: {waveform.shape[0]}, Duration: {waveform.shape[1]/sample_rate} sec")