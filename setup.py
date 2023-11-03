from setuptools import setup
setup(
    packages=["dismal_random_sims"],
    name='dismal_random_sims',
    install_requires=[
        "dismal @ git+https://github.com/simonharnqvist/DISMaL.git#egg=dismal",
        "pyyaml",
        "numpy",
        "scipy",
        "msprime",
        "ruffus",
        "joblib",
        "tqdm"],
    entry_points={
        'console_scripts': [
            'dismal-random-sims=dismal_random_sims.random_simulation_pipeline:main'
        ]
    }
)