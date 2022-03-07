import argparse

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import sys 

if(sys.version_info[0] < 3):
    logger.error("python version 3.x required ! ")

def parse_OCP_script(argv=None):
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--robot_name", type=str, default='iiwa', help="Name of the robot")
    PARSER.add_argument('--PLOT', action='store_true', default=False, help="Plot OCP solution")
    PARSER.add_argument('--VISUALIZE', action='store_true', default=False, help="Animate solution in Gepetto Viewer")
    return PARSER.parse_args(argv)


def parse_MPC_script(argv=None):
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--robot_name", type=str, default='iiwa', help="Name of the robot")
    PARSER.add_argument('--simulator', type=str, default='bullet', help="Name of the simulator")
    PARSER.add_argument('--PLOT_INIT', action='store_true', default=False, help="Plot warm-start solution")
    return PARSER.parse_args(argv)