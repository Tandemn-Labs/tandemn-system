# Orca

Orca is Tandemn's system to orchestrate accelerator workloads across a cluster of resources in order to improve utilization.

*Note: Orca is under active development and there will be frequent breaking changes to code and architecture*


## Architecture

A central server is the main interchange between the outside world and jobs running in the compute nodes controlled in the cluster. Each compute node in turns runs a service that listens to deployment and run commands from the central.

Every time a user request is submitted, an orchestrator software looks at the state of the resources in the world and makes decisions about the compute nodes to use to launch the job as well as the specific launch and job configurations.

Orchestrator makes decisions about what to run where, and the central server ingests these decisions, using them to control the compute nodes, The compute nodes then carries out the real heavy lifting of computation

### Overview of folders and files
- central_server/ contains programs that control the system
  - central_server.py is a FastAPI webserver that multiplexes between orchestrator, compute nodes and the outside world
  - magic.py is a placeholder for an eventually smarter orchestrator. It contains some basic code to illustrate the input and output formats
  - models/models.py contains legacy model definitions. This file is deprecated and message definitions are being migrated to ProtoBuf
- node/ contains programs that control the individual compute nodes
  - node_launch.py runs a small service that listens for commands from the central server and runs the workload accordingly
- models/ contains the ProtoBuf definition files as well as the corresponding Python bindings

### Tech stack
- Redis is used as the queue for user request coming in as well as deployment decisions made by the orchestrator that is awaiting action by central server
- MongoDB is used as the backing store for most metadata, such as job info and deployment info
- ProtoBuf is used as the binary serialization format for messages sent over the wire
- ZMQ is used for the communication between compute nodes 


## Installation

*Note: Assuming EC2 Ubuntu instances. Environment variables are used to capture most ephemeral information, like port numbers, node names and ip addresses*

### Compute nodes

Make sure Nvidia drivers are installed (vLLM sometimes require `python3-dev`):

```
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit python3-dev
```

Set up Python virtual environment: \
*Note: You may need to add uv to path after installing it*
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r node/requirements.txt 
```

Populate environment variables. Create a `.env` file and write the following variables:
```
TD_HOSTNAME= # Use a human readable hostname
TD_LISTENING_PORT=12345
TD_DEVICE_TYPE= # e.g. "nvidia-v100-32"
TD_NUM_DEVICES= # e.g. 4 GPUs will be "4" 
TD_CENTRAL_ADDR= # Address of central server
TD_CENTRAL_PORT= # Port of central server
STORAGE_SERVER_URL= # Address of storage server
STORAGE_SERVER_PORT= # Port of storage server
```

Finally, run the service on the node that listens for commands and do work:
```
python -m node.node_launch
```

### Central server

Make sure Redis container is up:
``` 
docker run -d --name tandemn-redis -p 6379:6379 --restart unless-stopped redis:7-alpine 
```

Using hosted MongoDB for now, make sure URI is populated in .env:
```
MONGODB_URI=mongodb+srv://... # Fill in with the actual connection string
```

Run the central server:
```
python -m central_server.central_server 
```

### Orchestrator

Run the orchestrator (cane be run on the same node as central server):
```
python -m central_server.magic
```
