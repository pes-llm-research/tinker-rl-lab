#!/usr/bin/env python3
"""
vast.ai Runner for SkyRL tx (Tinker API Server)

Provisions GPU instances on vast.ai and runs SkyRL tx server,
allowing any Tinker cookbook script to connect remotely.

Usage:
    python -m skyrl.backends.vastai_runner \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --algorithm grpo \
        --epochs 20
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import asyncssh


@dataclass
class VastInstance:
    id: int
    ssh_host: str
    ssh_port: int
    gpu_name: str
    num_gpus: int
    price: float
    status: str


class VastAILauncher:
    """
    Launcher for SkyRL tx on vast.ai instances.

    This class:
    1. Searches for available GPU instances
    2. Provisions instances
    3. Installs SkyRL and dependencies
    4. Starts the SkyRL tx Tinker API server
    5. Runs the specified training script
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        instance_type: str = "a100-80gb",
        ssh_key_path: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("VAST_API_KEY")
        if not self.api_key:
            raise ValueError("vast.ai API key required. Set VAST_API_KEY env var.")

        self.instance_type = instance_type
        self.ssh_key_path = ssh_key_path or os.path.expanduser("~/.ssh/id_rsa")
        self.instances: List[VastInstance] = []

    def run_command(self, cmd: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a vast CLI command."""
        full_cmd = ["vast", *cmd]
        if "--api-key" not in full_cmd:
            full_cmd.extend(["--api-key", self.api_key])
        return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)

    async def search_instances(self, instance_type: str = "a100") -> List[Dict[str, Any]]:
        """Search for available instances."""
        print(f"Searching for {instance_type} instances...")
        result = self.run_command(["search", "instances", instance_type, "--json"])

        if result.returncode != 0:
            print(f"Search failed: {result.stderr}")
            return []

        try:
            instances = json.loads(result.stdout)
            return instances[:5]  # Top 5 cheapest
        except json.JSONDecodeError:
            print(f"Failed to parse results: {result.stdout[:500]}")
            return []

    async def launch_instance(self, instance_id: int) -> Optional[VastInstance]:
        """Launch a single instance."""
        print(f"Launching instance {instance_id}...")

        # Start instance
        result = self.run_command(
            ["create", "instance", str(instance_id), "--json"],
            timeout=120
        )

        if result.returncode != 0:
            print(f"Launch failed: {result.stderr}")
            return None

        try:
            info = json.loads(result.stdout)
            instance = VastInstance(
                id=info["id"],
                ssh_host=info["ssh_host"],
                ssh_port=info.get("ssh_port", 22),
                gpu_name=info.get("gpu_name", "unknown"),
                num_gpus=info.get("num_gpus", 1),
                price=info.get("dph_total", 0),
                status="starting",
            )
            print(f"  Launched instance {instance.id} on {instance.ssh_host}")
            return instance
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse instance info: {e}")
            return None

    async def wait_for_ready(self, instance: VastInstance, timeout: int = 300) -> bool:
        """Wait for instance to be SSH-ready."""
        print(f"  Waiting for instance {instance.id} to be ready...")

        start = time.time()
        while time.time() - start < timeout:
            try:
                async with asyncssh.connect(
                    instance.ssh_host,
                    port=instance.ssh_port,
                    username="root",
                    known_hosts=None,
                    server_host_key_algs=[],
                ) as conn:
                    result = await conn.run("echo ready", check=True)
                    if result.exit_status == 0:
                        instance.status = "ready"
                        print(f"  Instance {instance.id} is ready!")
                        return True
            except Exception:
                pass

            await asyncio.sleep(15)

        print(f"  Instance {instance.id} did not become ready in {timeout}s")
        return False

    async def setup_instance(self, instance: VastInstance, setup_script: str) -> bool:
        """Setup SkyRL on the instance."""
        print(f"  Setting up SkyRL on instance {instance.id}...")

        try:
            async with asyncssh.connect(
                instance.ssh_host,
                port=instance.ssh_port,
                username="root",
                known_hosts=None,
                server_host_key_algs=[],
            ) as conn:

                # Write setup script
                async with conn.start_sftp() as sftp:
                    await sftp.putfo(
                        setup_script.encode(),
                        "/root/setup_skyrl.sh"
                    )

                # Run setup
                result = await conn.run(
                    "chmod +x /root/setup_skyrl.sh && bash /root/setup_skyrl.sh",
                    timeout=600,  # 10 min timeout
                )

                if result.exit_status != 0:
                    print(f"  Setup failed: {result.stderr}")
                    return False

                print(f"  Setup complete on instance {instance.id}")
                return True

        except Exception as e:
            print(f"  Setup error: {e}")
            return False

    async def start_skyrl_server(
        self,
        instance: VastInstance,
        model_name: str,
        port: int = 8000,
        tensor_parallel_size: int = 1,
    ) -> bool:
        """Start the SkyRL tx Tinker API server."""
        print(f"  Starting SkyRL tx server on instance {instance.id}...")

        start_cmd = f'''
cd /root/SkyRL
source .venv/bin/activate

# Start server in background
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra gpu --extra tinker -m skyrl.tinker.api \
    --base-model {model_name} \
    --port {port} \
    --backend-config '{{"tensor_parallel_size": {tensor_parallel_size}, "max_lora_adapters": 3}}' \
    > /root/skyrl_server.log 2>&1 &

echo "Server starting, PID: $!"
sleep 5
cat /root/skyrl_server.log
'''

        try:
            async with asyncssh.connect(
                instance.ssh_host,
                port=instance.ssh_port,
                username="root",
                known_hosts=None,
                server_host_key_algs=[],
            ) as conn:

                result = await conn.run(start_cmd, timeout=30)

                if result.exit_status == 0:
                    print(f"  SkyRL tx server started on {instance.ssh_host}:{port}")
                    return True
                else:
                    print(f"  Failed to start server: {result.stderr}")
                    return False

        except Exception as e:
            print(f"  Start error: {e}")
            return False

    def generate_setup_script(
        self,
        model_name: str,
        algorithm: str = "grpo",
        training_script: Optional[str] = None,
    ) -> str:
        """Generate the instance setup script."""
        script = f'''#!/bin/bash
set -euo pipefail
exec > /root/setup.log 2>&1

echo "=== $(date) | Setting up SkyRL ==="

# System prep
apt-get update -qq
apt-get install -y -qq git curl wget build-essential

# NVIDIA drivers check
nvidia-smi || echo "No NVIDIA GPU detected"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone SkyRL
git clone --depth 1 --branch skyrl_train-v0.4.0 \\
    https://github.com/NovaSky-AI/SkyRL.git /root/SkyRL

cd /root/SkyRL/skyrl-train

# Create venv and install
uv venv --python 3.12 --seed
source .venv/bin/activate
uv sync --extra vllm --extra gpu --extra tinker

# Install additional deps
uv pip install wandb datasets math-verify latex2sympy2-extended trl peft accelerate

echo "=== Setup Complete ==="
'''
        return script

    async def run_training(
        self,
        instance: VastInstance,
        training_script: str,
        env_vars: Dict[str, str],
    ) -> bool:
        """Run training script on instance."""
        print(f"  Running training on instance {instance.id}...")

        # Prepare env vars
        env_setup = "\n".join([f'export {k}="{v}"' for k, v in env_vars.items()])

        cmd = f'''
{env_setup}

cd /root/SkyRL
source .venv/bin/activate

# Run training
bash -c '{training_script}'
'''

        try:
            async with asyncssh.connect(
                instance.ssh_host,
                port=instance.ssh_port,
                username="root",
                known_hosts=None,
                server_host_key_algs=[],
            ) as conn:

                result = await conn.run(cmd, timeout=3600)  # 1 hour max

                if result.exit_status == 0:
                    print(f"  Training completed successfully")
                    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                    return True
                else:
                    print(f"  Training failed: {result.stderr}")
                    return False

        except Exception as e:
            print(f"  Training error: {e}")
            return False

    async def cleanup_instance(self, instance: VastInstance):
        """Stop and destroy an instance."""
        print(f"  Cleaning up instance {instance.id}...")
        self.run_command(["destroy", "instance", str(instance.id)], timeout=60)

    async def run(
        self,
        model_name: str,
        algorithm: str = "grpo",
        epochs: int = 20,
        num_instances: int = 1,
        training_command: Optional[str] = None,
    ):
        """Main execution flow."""
        print(f"\n{'='*60}")
        print(f"  SkyRL vast.ai Launcher")
        print(f"  Model: {model_name}")
        print(f"  Algorithm: {algorithm}")
        print(f"  Instances: {num_instances}")
        print(f"{'='*60}\n")

        try:
            # 1. Search for instances
            available = await self.search_instances(self.instance_type)

            if not available:
                print("No suitable instances found. Trying broader search...")
                available = await self.search_instances("a100")

            if not available:
                print("ERROR: No instances available")
                return

            # 2. Launch instances
            for i, inst_info in enumerate(available[:num_instances]):
                instance = await self.launch_instance(inst_info["id"])
                if instance:
                    self.instances.append(instance)

            # 3. Wait for instances to be ready
            ready_instances = []
            for instance in self.instances:
                if await self.wait_for_ready(instance):
                    ready_instances.append(instance)

            if not ready_instances:
                print("ERROR: No instances became ready")
                return

            # 4. Setup SkyRL on each instance
            setup_script = self.generate_setup_script(model_name, algorithm)

            for instance in ready_instances:
                if not await self.setup_instance(instance, setup_script):
                    print(f"WARNING: Setup failed for instance {instance.id}")

            # 5. Start SkyRL tx server
            for instance in ready_instances:
                await self.start_skyrl_server(instance, model_name)

            # 6. Run training
            if training_command:
                env_vars = {
                    "TINKER_API_KEY": "tml-dummy",
                    "TINKER_BASE_URL": f"http://{instance.ssh_host}:8000",
                    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                }

                await self.run_training(instance, training_command, env_vars)

            print("\n" + "=" * 60)
            print("  All tasks completed!")
            print("=" * 60)

        finally:
            # Cleanup
            if input("Destroy instances? [y/N] ").lower() == "y":
                for instance in self.instances:
                    await self.cleanup_instance(instance)


def main():
    parser = argparse.ArgumentParser(description="Launch SkyRL on vast.ai")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model to use")
    parser.add_argument("--algorithm", type=str, default="grpo",
                        choices=["grpo", "ppo", "reinforce"],
                        help="RL algorithm")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--instance-type", type=str, default="a100-80gb",
                        help="vast.ai instance type")
    parser.add_argument("--num-instances", type=int, default=1,
                        help="Number of instances")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallelism size")
    parser.add_argument("--training-command", type=str, default=None,
                        help="Training command to run")
    parser.add_argument("--api-key", type=str, default=None,
                        help="vast.ai API key (or set VAST_API_KEY)")

    args = parser.parse_args()

    launcher = VastAILauncher(api_key=args.api_key, instance_type=args.instance_type)

    asyncio.run(launcher.run(
        model_name=args.model,
        algorithm=args.algorithm,
        epochs=args.epochs,
        num_instances=args.num_instances,
        training_command=args.training_command,
    ))


if __name__ == "__main__":
    main()
