Datasets:

experimental-slam-training-data1.npz

* size: 1000

* load/save via: `slam_data.save_dataset()` / `load_dataset()`

* format: `((input_map, lds_map), (ground_truth_map, adlo))`

* Generated via slam_data.generate_training_data(), with all 4 sample types. Uses floorplan crops for input maps, using only floorplan 2.

* Doesn't contain LDS noise.

experimental-slam-training-data2.npz

- size: 1000

- load/save via: `slam_data.save_dataset()` / `load_dataset()`

- format: `((input_map, lds_map), (ground_truth_map, adlo), metadata)`

- Generated via slam_data.generate_training_data(), with all 4 sample types. Uses floorplan crops for input maps, using only floorplan 2.

- Contains simulated LDS noise.

- Map and -DLO outputs accurately blanked out when they should be ignored from loss calculations.

experimental-slam-training-data3.npz

- size: 1000

- load/save via: `slam_data.save_dataset()` / `load_dataset()`

- format: `((input_map, lds_map), (ground_truth_map, adlo), metadata)`

- Generated via slam_data.generate_training_data(), with all 4 sample types. Uses floorplan crops for input maps, using only floorplan 2.

- Contains simulated LDS noise.

- Includes ground-truth output maps for all sample types.

- -DLO outputs accurately blanked out when they should be ignored from loss calculations.

- BUG: was generated with pixel_size=44.71 instead of 47.41.

experimental-slam-training-data4.npz

- size: 1000

- load/save via: `slam_data.save_dataset()` / `load_dataset()`

- format: `((input_map, lds_map), (ground_truth_map, adlo), metadata)`

- Generated via slam_data.generate_training_data(), with all 4 sample types. Uses floorplan crops for input maps, using only floorplan 2.

- Contains simulated LDS noise.

- Blanks out ground-truth output maps for sample type 2 only.

- -DLO outputs accurately blanked out when they should be ignored from loss calculations.

- BUG: was generated with pixel_size=44.71, resulting in incorrect window sizes of 159x159x3. Should have been pixel_size 47.41 and window size 149x149x3.

experimental-slam-training-data5.npz

* Same as v4 but pixel_size fixed (now 47.41), so that window sizes are 149x149x3.

slam-training-data.npz

- size: 1000

- load/save via: `slam_data.save_dataset()` / `load_dataset()`

- format: `((input_map, lds_map), (ground_truth_map, adlo), metadata)`

- semantic map size: 149x149x3

- Generated from "experimental floorplan 2 only.

- Generated via slam_data.generate_training_data(), with all 4 sample types. Uses floorplan crops for input maps. Contains simulated LDS noise. Ground-truth output maps blanked (all pixels 'unknown') for sample type 2 only. DLO outputs blanked out (via NANs) when they should be ignored from loss calculations.

- BUG FIX: Includes fixes to sub-pixel alignment problems in LDS and semantic map samples that seeded wall-sliding effect when agent navigated around environments.
