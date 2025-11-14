# Tandemn-orca

#############################
Server side- 
1 - Install central_server requirements (uv pip install -r server_requirements.txt)
2 - Run docker image for redis

docker run -d \
  --name tandemn-redis \
  -p 6379:6379 \
  --restart unless-stopped \
  redis:7-alpine

# verify
docker ps | grep redis
redis-cli ping

# visualize
docker run -d \
  --name redis-commander \
  -p 8081:8081 \
  --env REDIS_HOSTS=local:172.17.0.1:6379 \
  --restart unless-stopped \
  ghcr.io/joeferner/redis-commander:latest
  
TODO - Combine them in a single docker image for easy installation.
############################

############################
Node side - 
1 - Install the node requirements (uv pip install -r node/node_requirements.txt)
2 - python -m node/machine_runner.py (it should run in the base os and not in a container)
3 - Signals into this script, leads to launching the environments.
############################