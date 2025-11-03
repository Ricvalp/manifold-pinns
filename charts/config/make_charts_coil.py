from datetime import datetime

from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 37
    cfg.figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # #  Dataset  # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.dataset = ConfigDict()
    cfg.dataset.seed = 37
    cfg.dataset.name = "Coil"
    cfg.dataset.path = "./datasets/obj_files/coil.obj"
    cfg.dataset.scale = 1.0
    cfg.dataset.points_per_unit_area = 8
    cfg.dataset.subset_cardinality = None
    cfg.dataset.charts_path = "./datasets/coil/charts_1"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #   Charts  # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cfg.charts = ConfigDict()
    cfg.charts.alg = "fast_region_growing"
    cfg.charts.min_dist = 9.
    cfg.charts.nearest_neighbors = 10
    cfg.charts.deformation_magnitude = 0.1

    return cfg
