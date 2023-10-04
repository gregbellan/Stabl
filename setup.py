import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='Stabl',
    version='0.0.1',
    author='Grégoire Bellan',
    author_email='gbellan@surge.care',
    description='Stabl package',
    packages=['stabl'],
    install_requires=[
        'joblib==1.1.0',
        'tqdm==4.64.0',
        'matplotlib==3.5.2',
        "knockpy==1.2",
        "scikit-learn==1.1.2",
        "seaborn==0.12.0",
        "groupyr==0.3.2",
        "pandas==1.4.2",
        "statsmodels==0.14.0",
        "openpyxl==3.0.7",
        "adjustText==0.8",
        "scipy==1.10.1",
        "julia==0.6.1",
        "osqp==0.6.2",
    ]
)
