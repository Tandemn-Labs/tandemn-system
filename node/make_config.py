#!/usr/bin/env python3
"""
Tandemn Node Configuration Generator
This script creates a .env file with all necessary configuration for:
- machine_runner.py
- node_health_agent.py
"""

import socket
import uuid
import os
from pathlib import Path


def get_user_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        while True:
            user_input = input(f"{prompt}: ").strip()
            if user_input:
                return user_input
            print("⚠️  This field is required. Please provide a value.")


def generate_config():
    """Generate .env configuration file for Tandemn node services."""
    
    print("=" * 60)
    print("  Tandemn Node Configuration Generator")
    print("=" * 60)
    
    # Auto-detect defaults
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("\n Central Server Configuration:")
    central_server_host = get_user_input(
        "Central Server IP/Hostname",
        default="localhost"
    )
    central_server_port = get_user_input(
        "Central Server Port",
        default="8000"
    )
    print("\n Machine Runner Configuration:")
    machine_runner_port = get_user_input(
        "   Machine Runner Port",
        default="8001"
    )
    
    # Generate a unique but consistent NODE_ID
    node_id = f"{hostname}-{str(uuid.uuid4())}"
    
    # Health monitoring configuration
    print("\n Health Monitoring Configuration:")
    health_check_interval = get_user_input(
        "Health check interval (seconds)",
        default="2.0"
    )
    
    # Build configuration
    config = {
        "CENTRAL_SERVER_HOST": central_server_host,
        "CENTRAL_SERVER_PORT": central_server_port,
        "MACHINE_RUNNER_HOST": "localhost",  # Health agent talks to local machine runner
        "MACHINE_RUNNER_PORT": machine_runner_port,
        "NODE_ID": node_id,
        "IP_ADDRESS": local_ip,
        "MACHINE_RUNNER_URL": f"http://{local_ip}:{machine_runner_port}",
        "HEALTH_CHECK_INTERVAL": health_check_interval,
    }
    
    # Write .env file
    env_path = Path(__file__).parent / ".env"
    
    with open(env_path, 'w') as f:     
        f.write("# Central Server Configuration\n")
        f.write(f"CENTRAL_SERVER_HOST={config['CENTRAL_SERVER_HOST']}\n")
        f.write(f"CENTRAL_SERVER_PORT={config['CENTRAL_SERVER_PORT']}\n")
        f.write("\n")
        f.write("# Machine Runner Configuration\n")
        f.write(f"MACHINE_RUNNER_HOST={config['MACHINE_RUNNER_HOST']}\n")
        f.write(f"MACHINE_RUNNER_PORT={config['MACHINE_RUNNER_PORT']}\n")
        f.write("\n")
        f.write("# Node Identification\n")
        f.write(f"NODE_ID={config['NODE_ID']}\n")
        f.write(f"IP_ADDRESS={config['IP_ADDRESS']}\n")
        f.write(f"MACHINE_RUNNER_URL={config['MACHINE_RUNNER_URL']}\n")
        f.write("\n")
        f.write("# Health Monitoring\n")
        f.write(f"HEALTH_CHECK_INTERVAL={config['HEALTH_CHECK_INTERVAL']}\n")
    return True


def main():
    """Main entry point."""
    generate_config()



if __name__ == "__main__":
    exit(main())

