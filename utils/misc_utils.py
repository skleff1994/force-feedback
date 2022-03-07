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
    PARSER.add_argument('--PLOT', action='store_true', default=False)
    PARSER.add_argument('--VISUALIZE', action='store_true', default=False)
    return PARSER.parse_args(argv)


def parse_MPC_script(argv=None):
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--robot_name", type=str, default='iiwa', help="Name of the robot")
    PARSER.add_argument('--PLOT', action='store_true', default=False)
    PARSER.add_argument('--VISUALIZE', action='store_true', default=False)
    return PARSER.parse_args(argv)