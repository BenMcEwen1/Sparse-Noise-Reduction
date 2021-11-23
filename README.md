# Annotator

## Description
Annotation tool to recommend regions of recordings likely to contain target audio events
This method uses wavelet Packet Decomposition for noise reduction and compares to reference packets

## Things to do
- [x] Comparison between recording spectrograms and reference packets
  
- [x] Correlation between multiple reference recordings
  - [x] Store references as .npy arrays so they dont need to be converted each time
- [x] Standardise shape of recommendations - Can be improved further
- [x] Rank recommendations in order of highest correlation
- [x] Bug fix: case where there is an od number of time stamps

- [x] Generate more masks (unique call types and noise)
- [ ] Integrate with wavelet denoising 
- [x] convert samples to timestamp to segment real time-domain signal
- [ ] Check for overlapping recommendation and combine
  - [x] Combine similar time stamps
  - [ ] Combine call and correlation labels relating to time stamp