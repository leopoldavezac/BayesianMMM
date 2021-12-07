from setuptools import setup, find_packages

setup(
    name= 'bayesian_mmm',
    version= '1.0',
    description= "an implementation of bayesian media mix modelling",
    url= "https://github.com/leopoldavezac/BayesianMMM",
    author= "Leopold Davezac",
    author_email= "leopoldavezac@gmail.com",
    license= "MIT",
    keywords= "bayesian media mix modeling stan sampling carryover shape effect",
    packages= find_packages(exclude=("tests", "docs")),
    python_requires= ">=3.7",
    entry_points= {
    'console_scripts': [
        'train_bayesian_mmm=bayesian_mmm.train:run',
        ]
    }
)