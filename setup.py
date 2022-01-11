from setuptools import setup, find_packages
  

setup(
    name="mlflow-kernel",
    version="0.1.10",
    description="MLFlow kernel for jupyter",
    packages=find_packages(),
    package_data={'mlflow_kernel': ['kernel.json']},
    install_requires=["infinstor_mlflow_plugin", "ipykernel>=6.0.0", "jupyter_client<=6.99.99", "ipython_genutils", "boto3"]
)
