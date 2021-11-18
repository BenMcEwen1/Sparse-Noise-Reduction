# Annotator

## Description
Annotation tool to recommend regions of recordings likely to contain target audio events
This method uses wavelet Packet Decomposition for noise reduction and compares to reference packets

## Things to do
- [x] Comparison between recording spectrograms and reference packets
  
- [ ] Correlation between multiple reference recordings
  - [ ] Store references as .npy arrays so they dont need to be converted each time
- [ ] Standardise shape of recommendations
- [ ] Rank recommendations in order of highest correlation
- [ ] Bug fix: case where there is an od number of time stamps