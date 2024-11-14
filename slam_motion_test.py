from slam_motion import *


def run_test_suite():
    get_contour_pxcoords_test()


def get_contour_pxcoords_test():
    trajectory = get_contour_pxcoords('repo/data/experimental-floorplan2-with-trajectory.png')
    assert trajectory.ndim == 2
    assert trajectory.shape[1] == 2
