from mlflow_kernel.mlflow_kernel import MLFlowKernel

if __name__ == '__main__':
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=MLFlowKernel)
