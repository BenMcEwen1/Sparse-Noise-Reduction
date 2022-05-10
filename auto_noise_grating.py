from spectral_noise_grating import spectrogram, autoThreshold, mask, reconstruct, plot_spectrogram, ISTFT, convertToAmp
import librosa
import soundfile as sf
import time

filename = 'field16k'

recording, sample_rate = librosa.load(f'./Sound Files/{filename}.wav', sr=None)
print(sample_rate)

start = time.time()
sig_stft, sig_stft_db = spectrogram(recording)
thresh = autoThreshold(sig_stft_db)
masked, mask = mask(thresh, sig_stft, sig_stft_db)
reconstructed, original = reconstruct(masked, recording)

denoised = convertToAmp(reconstructed)
denoised = ISTFT(denoised)
end = time.time()

diff = end - start
print(diff)

sf.write(f'./denoised/{filename}_denoised.wav', denoised, sample_rate)

# Original and denoised
# plot_spectrogram(original, title="Original spectrogram")
# plot_spectrogram(reconstructed, title="Reconstructed spectrogram")