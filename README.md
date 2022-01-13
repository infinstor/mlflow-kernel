# MLflow Kernel

  MLflow Kernel connects to an MLflow service and records all the data science activities in the notebook. MLflow Kernel captures the cell code, the cell output and the visualizations, and records them as mlflow artifacts.
  
  Each cell execution is captured as a separate run. This happens transparently, and the user doesn't have to manage the lifecycle of a run. All the runs of these cells are grouped inside a parent run that stays alive until the life of the kernel. A new parent run is created when the kernel is restarted.
  
  The kernel works seamlessly with the autologging feature which is supported by many ML libraries. The user just needs to enable autologging and everything is recorded as part of cell execution run.
  
  For more information, visit https://www.infinstor.com/quickstart/mlflow-kernel.

## Steps to Deploy MLflow Kernel

1. pip install mlflow_kernel
    - This will install all required packages
    - This will also setup the kernelspecs
2. Configuration: Write [HOME]/.jupyter/mlflow_kernel_config.json
    - [HOME] is the home directory of the user that starts the kernel. It is usually the same as the user that starts the notebook server.
    - The content of the config file is as follows:
      >{ <br>
      >   "mlflow_tracking_uri": [mlflow tracking url],<br>
      >   "debug_enabled" : "false",<br>
      >   "default_experiment_name": [experiment name]<br>
      >}
    - The "mlflow_tracking_uri" points to the mlflow server, for example if you are using infinstor’s mlflow service, the mlflow_tracking_uri would be infinstor://mlflow.infinstor.com
    - If default_experiment_name is not specified, the runs are created in 'default' experiment with id 0. The user should be authorized for the specified experiment or for the 'default' experiment.


## MLflow Kernel with Infinstor’s MLflow service
  Following simple steps will get the mlflow kernel up and connected to the infinstor’s mlflow service.
1. In the mlflow_kernel_config.json as described in the previous section, set the mlflow_tracking_uri to point to the Infinstor’s service.
2. First time access on the notebook will prompt the user to perform the mlflow login. This will require user's infinstor username and password. Login is required for the first time only, and afterwards tokens are automatically refreshed until refreshtoken expires. The kernel must be restarted after the login.
  - mlflow login can be performed using following two lines of code from any cell

            import infinstor_mlflow_plugin  
            infinstor_mlflow_plugin.login.new_login()


## MLflow Kernel with Sagemaker Studio
  MLflow kernel can be used with sagemaker studio using Infinstor’s pre-built image. This image is preconfigured to connect to Infinstor’s MLflow service. The image uri is *986605205451.dkr.ecr.us-east-1.amazonaws.com/mlflow_kernel:mlflow_kernel*
  
  Steps:
  - Please follow AWS documentation at  https://docs.aws.amazon.com/sagemaker/latest/dg/studio-byoi-attach.html to set up sagemaker studio with the above mentioned kernel image uri.
  - The role attached with the kernel, must have access to the mlflow's artifact bucket. The default artifact bucket can be found in the user's profile on the infinstor's service dashboard.
  - First time access on the notebook will prompt the user to perform the mlflow login. This will require user's mlflow login and password. Login is required for the first time only, and afterwards tokens are automatically refreshed. The kernel must be restarted after the login.
  - mlflow login can be performed using following two lines of code from any cell
  
            import infinstor_mlflow_plugin  
            infinstor_mlflow_plugin.login.new_login()

