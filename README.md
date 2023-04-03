# Automatic Noise Reduction of Extremely Sparse Vocalisations for Bioacoustic Monitoring

## Abstract
Environmental noise and data sparsity present major challenges within the field of bioacoustics. This paper explores noise reduction (audio enhancement) techniques in the context of extremely sparse vocalisations (<1% occurrence rates) of invasive mammalian and marsupial species and the clear implications for other bioacoustics applications which face similar challenges. We compare relevant noise reduction techniques and recommend a spectral subtraction approach that outperforms alternative approaches in terms of stationary noise reduction, efficiency, and data requirements. We identify how the contributions of this work can be applied within the broader context of bioacoustics. We also explore the current benefits and limitations of state-of-the-art deep audio enhancement approaches within the context of bioacoustics applications.

## Installation
```pip install -r requirements.txt```

## Description
Repository accompanying paper (*Automatic Noise Reduction of Extremely Sparse Vocalisations for Bioacoustic Monitoring*). Includes the New Zealand (NZ) Invasive Predator test dataset `/audio/predator`. Find NZ Native Bird dataset at: https://github.com/smarsland/AviaNZ or `/audio/bird`

### Results
Full results (.wav) can be found at `/results`, this includes Wavelet Packet Decomposition (WPD), spectral subtraction, CMGAN (spectral) and CMGAN (additive) for both datasets.

### Evaluation
SnNR, Success Ratio (SR) and PSNR metric can be generated using `evaluate.py`. Select results to evaluate from `/results` and the corresponding dataset (.json), either `Bird_dataset.json` or `Predator_dataset.json`

### Feature Spacing
Find feature spacing test audio at `/feature_spacing`. This is evaluated using `manual_evaluation.py`

## Citation
Coming soon

## Acknowledgements
This research was made possible through Capability Development funding through Predator Free 2050 Limited (https://pf2050.co.nz/).
AviaNZ (https://github.com/smarsland/AviaNZ) for use of their WPD method and NZ native bird dataset.
Tim Sainbury (https://github.com/timsainb/noisereduce) for use of the noisereduce package (spectral subtraction)
Cao et al. (https://github.com/ruizhecao96/CMGAN) for access to CMGAN.