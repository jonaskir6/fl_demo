# Demonstration of Federated Learning with simple Classification Tasks

## Requirements
Runs on **Linux**
- Python: https://www.python.org/downloads/
- Docker with the NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Running the Federated Learning Demo
The folders with demo_* represent the different versions of the demos, choose a version and run the test with your own model and data or run the basic demo

### Run the customizable demo with your own model and data

1. Add your custom model code to `demo_custom/client/model.py` & `demo_custom/server/model.py`
3. Add code for loading your data inside `demo_custom/client/client.py`, follow instructions in file
4. Add your data to a folder in `demo_custom` and specify the data folders with `client_x` for the xth client
5. Specify your configuration using the `config.json` file:  
    - num_clients: the number of clients
    - source_path: your data source path
    - rounds: the number of exchange rounds (iterations)
    - num_epochs: number of train epochs
    - batch_size: batch sizes for MNIST data
    - learning_rate: learning rate for the model
6. Run the *script.py*  
    ```
    python3 run.py
    ```


### Run the basic demo

1. Specify your configuration using the `config.json` file:  
    - num_clients: the number of clients
    - rounds: the number of exchange rounds (iterations)
    - num_epochs: number of train epochs
    - batch_size: batch sizes for MNIST data
    - learning_rate: learning rate for the model
2. Run the *script.py*  
    ```
    python3 run.py
    ```


