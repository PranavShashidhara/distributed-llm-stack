import os
import time
import runpod
import subprocess
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")
os.environ["FORCE_IPV4"] = "1"
def select_gpu(preferred_display_name="RTX 3090"):
    """Finds a valid GPU ID from RunPod matching the preferred display name."""
    gpus = runpod.get_gpus()
    for g in gpus:
        if preferred_display_name in g['displayName']:
            return g['id']
    raise ValueError(f"No GPU found matching '{preferred_display_name}'")

def start_cluster(gpu_count=4, pod_name="SQL-Genie-Cluster", gpu_display_name="RTX A4500"):
    """Triggers a pod creation and returns the ID immediately."""
    gpu_type_id = select_gpu(gpu_display_name)
    print(f"Triggering {gpu_count}x {gpu_display_name}...")
    
    pod_response = runpod.create_pod(
        name=pod_name,
        image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        gpu_type_id=gpu_type_id,
        gpu_count=gpu_count,
        volume_in_gb=100,
        container_disk_in_gb=50 # Highly recommended for LLM training
    )

    # In the current SDK, pod_response is a dictionary with the ID
    pod_id = pod_response.get('id')
    print(f"Pod {pod_id} is now provisioning. Check your dashboard for the IP.")
    return pod_id


def stop_cluster(pod_name="SQL-Genie-Cluster"):
    """Terminates all pods with the specified name."""
    pods = runpod.get_pods()
    for pod in pods:
        if pod.get('name') == pod_name:
            print(f"Terminating pod {pod['id']}...")
            runpod.terminate_pod(pod['id'])

if __name__ == "__main__":
    # List available GPUs
    gpus = runpod.get_gpus()
    print("Available GPUs:")
    for g in gpus:
        print(f"- {g['displayName']} (ID: {g['id']}, Memory: {g['memoryInGb']} GB)")

    # Start the cluster
    #cluster_info = start_cluster(gpu_count=1)
    #print(f"Cluster started with ID: {cluster_info}")
    stop_cluster()           # optional cleanup