from slam_operations import *


def run_test_suite():
    get_contour_pxcoords_test()
    print("All slam_operations tests passed.")


def get_contour_pxcoords_test():
    trajectory = load_trajectory_pxcoords('repo/data/experimental-floorplan2-with-trajectory.png')
    assert trajectory.ndim == 2
    assert trajectory.shape[1] == 2
