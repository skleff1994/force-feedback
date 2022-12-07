from setuptools import setup, find_packages
from os import path, walk

package_name = 'force_feedback'

scripts_list = []
for (root, _, files) in walk(path.join("demos")):
    for demo_file in files:
        if('yml' not in demo_file):
            scripts_list.append(path.join(root, demo_file))
print(find_packages(include=['core_mpc', 
                             'soft_mpc', 
                             'classical_mpc', 
                             'lpf_mpc']))
setup(
    name=package_name,
    version="1.0.0",
    package_dir={
        "": "python",
    },
    packages=find_packages(where="python"),
    install_requires=["setuptools", 
                      "pybullet", 
                      "importlib_resources",
                      "bullet_utils",
                      "robot_properties_kuka"],
    zip_safe=True,
    maintainer="skleff",
    maintainer_email="sk8001@nyu.edu",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    description="MPC classical, force feedback, soft contact.",
    license="BSD-3-clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-clause",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    # cmdclass={"build_py": custom_build_py},
)
