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
