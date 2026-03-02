import os
import time
import runpod
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")
os.environ["FORCE_IPV4"] = "1"

POD_NAME       = "SQL-Genie-Inference"
GPU_NAME       = "RTX 5090"       # Change to "RTX 4090" or "A6000" if 5090 unavailable
GPU_COUNT      = 1                # TP=2 requires exactly 2 GPUs
DOCKER_IMAGE   = "vllm/vllm-openai:latest"
VOLUME_GB      = 100              # For model weights storage
DISK_GB        = 50
VLLM_PORT      = 8000

# ── GPU Selection ─────────────────────────────────────────────────────────────
def select_gpu(preferred_display_name=GPU_NAME):
    """Finds a GPU ID from RunPod matching the preferred display name."""
    gpus = runpod.get_gpus()
    print("\nAvailable GPUs:")
    for g in gpus:
        print(f"  - {g['displayName']} (ID: {g['id']}, VRAM: {g['memoryInGb']} GB)")

    for g in gpus:
        if preferred_display_name in g['displayName']:
            return g['id']

    # Fallback: find any GPU with >= 24GB VRAM
    print(f"\n '{preferred_display_name}' not found. Falling back to any GPU >= 24GB VRAM...")
    for g in gpus:
        if g['memoryInGb'] >= 24:
            print(f"  Using: {g['displayName']}")
            return g['id']

    raise ValueError("No suitable GPU found. Check RunPod availability.")


# ── Start Inference Cluster ───────────────────────────────────────────────────
def start_inference_cluster(
    model_path="/workspace/model",   # Path inside the pod where model is stored
    gpu_count=GPU_COUNT,
    pod_name=POD_NAME,
    gpu_display_name=GPU_NAME,
):
    """
    Provisions a 2x RTX 5090 RunPod pod and launches vLLM with:
      - Tensor Parallelism (TP=2)
      - Speculative Decoding (Llama 3.2 1B draft)
      - FP8 KV-Cache for 128k context
    """
    gpu_type_id = select_gpu(gpu_display_name)

    # vLLM launch command injected as the pod's startup command
    vllm_cmd = (
        f"vllm serve {model_path} "
        f"--tensor-parallel-size {gpu_count} "
        f"--speculative-model meta-llama/Llama-3.2-1B "
        f"--num-speculative-tokens 5 "
        f"--kv-cache-dtype fp8 "
        f"--max-model-len 131072 "          # 128k context window
        f"--gpu-memory-utilization 0.92 "   # leave 8% headroom
        f"--enable-prefix-caching "         # cache repeated SQL schema prefixes
        f"--port {VLLM_PORT} "
        f"--host 0.0.0.0"
    )

    print(f"\n Provisioning {gpu_count}x {gpu_display_name} pod: '{pod_name}'...")
    print(f"   vLLM command: {vllm_cmd}\n")

    pod_response = runpod.create_pod(
        name=pod_name,
        image_name=DOCKER_IMAGE,
        gpu_type_id=gpu_type_id,
        gpu_count=gpu_count,
        volume_in_gb=VOLUME_GB,
        container_disk_in_gb=DISK_GB,
        ports=f"{VLLM_PORT}/http",          # Expose vLLM port
        docker_args=vllm_cmd,
        env={
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )

    pod_id = pod_response.get('id')
    print(f" Pod '{pod_id}' is provisioning.")
    print(f"   Check your RunPod dashboard for the assigned IP.")
    print(f"   Once running, vLLM will be available at: http://<POD_IP>:{VLLM_PORT}")
    return pod_id


# ── Wait for Pod to be Ready ──────────────────────────────────────────────────
def wait_for_pod(pod_id, timeout_seconds=300):
    """Polls until the pod is RUNNING or timeout is reached."""
    print(f"\n Waiting for pod {pod_id} to start (timeout: {timeout_seconds}s)...")
    start = time.time()

    while time.time() - start < timeout_seconds:
        pods = runpod.get_pods()
        for pod in pods:
            if pod['id'] == pod_id:
                status = pod.get('desiredStatus', 'UNKNOWN')
                runtime = pod.get('runtime')
                if status == 'RUNNING' and runtime:
                    ip = runtime.get('ports', [{}])[0].get('ip', 'unknown')
                    print(f" Pod is RUNNING — IP: {ip}:{VLLM_PORT}")
                    return ip
                else:
                    print(f"   Status: {status} — waiting...")
        time.sleep(10)

    print(f"  Timeout reached. Check your RunPod dashboard manually.")
    return None


# ── Stop Cluster ──────────────────────────────────────────────────────────────
def stop_cluster(pod_name=POD_NAME):
    """Terminates all pods matching the given name."""
    pods = runpod.get_pods()
    found = False
    for pod in pods:
        if pod.get('name') == pod_name:
            print(f" Terminating pod {pod['id']}...")
            runpod.terminate_pod(pod['id'])
            found = True
    if not found:
        print(f"No pods found with name '{pod_name}'.")


# ── List All Pods ─────────────────────────────────────────────────────────────
def list_pods():
    """Prints a summary of all active RunPod pods."""
    pods = runpod.get_pods()
    if not pods:
        print("No active pods.")
        return
    print(f"\n{'ID':<20} {'Name':<25} {'Status':<12} {'GPU'}")
    print("-" * 75)
    for pod in pods:
        print(
            f"{pod.get('id',''):<20} "
            f"{pod.get('name',''):<25} "
            f"{pod.get('desiredStatus',''):<12} "
            f"{pod.get('machine', {}).get('gpuDisplayName', 'unknown')}"
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SQL-Genie RunPod Cluster Manager")
    parser.add_argument("action", choices=["start", "stop", "list", "gpus"],
                        help="Action to perform")
    parser.add_argument("--model-path", default="/workspace/model",
                        help="Path to model inside the pod (default: /workspace/model)")
    args = parser.parse_args()

    if args.action == "start":
        pod_id = start_inference_cluster(model_path=args.model_path)
        wait_for_pod(pod_id)

    elif args.action == "stop":
        stop_cluster()

    elif args.action == "list":
        list_pods()

    elif args.action == "gpus":
        gpus = runpod.get_gpus()
        print(f"\n{'Display Name':<35} {'ID':<25} {'VRAM (GB)'}")
        print("-" * 70)
        for g in gpus:
            print(f"{g['displayName']:<35} {g['id']:<25} {g['memoryInGb']}")
