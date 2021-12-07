# Annotator

## Description
Annotation tool to recommend regions of recordings likely to contain target audio events
This method uses wavelet Packet Decomposition for noise reduction and compares to reference packets


## Things to do
Pre-processing
- [x] Run initial checks: sampling rate = 16kHz, monochannel, normalised (maybe)

Classification
- [ ] Deep Learing Pipeline (Segmentation -> classifier)
- [ ] Transfer learning (Yamnet)
- [ ] Data augmentation of training set


## Complete
- [x] Comparison between recording spectrograms and reference packets
  
- [x] Correlation between multiple reference recordings
  - [x] Store references as .npy arrays so they dont need to be converted each time
- [x] Standardise shape of recommendations - Can be improved further
- [x] Rank recommendations in order of highest correlation
- [x] Bug fix: case where there is an od number of time stamps

- [x] Generate more masks (unique call types and noise)
- [x] Integrate with wavelet denoising 
- [x] convert samples to timestamp to segment real time-domain signal
- [x] Check for overlapping recommendation and combine
  - [x] Combine similar time stamps
  - [x] Combine call and correlation labels relating to time stamp