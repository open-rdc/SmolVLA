from setuptools import find_packages, setup


package_name = "smolvla_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/smolvla_nav.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="orne",
    maintainer_email="kasaiatsuki@gmail.com",
    description="SmolVLA inference navigation node + topological-map place recognition for ROS2 deployment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "navigation_node = smolvla_nav.navigation:main",
            "place_prompt_node = smolvla_nav.place_prompt_node:main",
            "create_topomap = smolvla_nav.create_topomap:main",
        ],
    },
)
