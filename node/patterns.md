# Async Patterns

Overall big event loop

Note: Need to check zmq.asyncio waiting pattern on sends. Does it return if 
socket is not connected

### Untracked coros
- `listen_to_central` uses an infinite loop to listen to central server, it does
`await` on the `launch_x` commands, but those commands shouldn't block, and will
return after setting up the model + cluster

### vLLM on Ray
- After worker nodes join Ray cluster, an untracked coro is created that listens
to commands from the head node

# Comms Patterns

### vLLM on Ray

All nodes receive `launch` command from central server. 
- ZMQ REQ/REP - Strict alternation # TODO: Look into non-strict patterns
- First node is head node. Sends start and Ray cluster listening port to all workers
- Workers reply once it has joined Ray cluster. (**Thing is, workers start listening
with timeout for msg from head once it receives `launch` command from central,
this is probably problematic because it can timeout prematurely, #TODO)
- The sockets that head node first used are saved in a list, reused to send
commands moving forward, worker nodes listen in a loop