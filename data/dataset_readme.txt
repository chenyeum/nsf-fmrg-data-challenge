NSF Future Manufacturing Data Challenge Dataset

This dataset contains multimodal data for probabilistic local geometry prediction in single DED laser tracks.

Modalities:
1. Thermal image sequences from a Stratonics ThermaViz melt-pool sensor.
2. SEM images from a Zeiss EVO MA10 system.
3. Full-field height maps from a Bruker ContourGT-K white-light 3D optical profilometer.

Physical conventions:
- Common analysis window: 20–100 mm.
- Thermal frame size: 400 × 400 pixels.
- Thermal pixel size: approximately 14 µm/pixel.
- Thermal field of view: approximately 5.6 mm × 5.6 mm.
- Thermal frame rate: 50 fps.
- Scan speed: 10 mm/s.
- Consecutive thermal frames correspond to approximately 0.2 mm of laser travel.
- Bruker/Wyko ASC files store x and y in mm and z in nm.
- Raw ASC local x = 0 corresponds to the physical 100 mm side; starter code maps height maps to increasing 20–100 mm actual coordinate order.
- SEM tile 01 corresponds to the 100 mm side; the highest-numbered SEM tile corresponds to the 20 mm side.

Acknowledgment:
This competition and associated material are based upon work supported by the National Science Foundation under Grant Number FMRG-2328395.

Citation:
Please cite the companion dataset paper and GitHub repository for any use outside the NSF Future Manufacturing Data Challenge.