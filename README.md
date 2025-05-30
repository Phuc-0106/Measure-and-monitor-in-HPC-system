GPU Monitoring & Energy Profiling Suite

This repository contains two complementary tools for monitoring NVIDIA GPU metrics and profiling ML training energy:

    gpu_nvml (C + NVML + libmicrohttpd)
    Exports per-GPU and per-process metrics (power, temperature, utilization, memory) on port 8080 in Prometheus format.

    train_resnet.py (Python + Zeus + Prometheus)
    Trains ResNet-18 on CIFAR-10 under various optimization strategies (baseline, mixed precision, larger batch, early stopping), exporting per-epoch GPU power and total energy on port 8000.

Table of Contents

    Prerequisites

    Building & Running gpu_nvml

    Setting Up & Running train_resnet.py

    Prometheus Configuration

    Usage Examples

Prerequisites

    NVIDIA GPU with a recent driver (supports NVML).

    Ubuntu 20.04+ or equivalent Linux.

    C toolchain: gcc, make, pkg-config.

    Libraries for C exporter:

sudo apt-get install libmicrohttpd-dev libnvidia-ml-dev

Python 3.8+, venv, and pip.

Zeus and Prometheus client for Python:

    python3 -m venv venv
    source venv/bin/activate
    pip install torch torchvision zeus prometheus_client

Building & Running gpu_nvml

    Clone or place gpu_nvml.c in your working directory.

    Compile:

gcc gpu_nvml.c -o gpu_nvml \
  -I/usr/include/nvidia-ml \
  -L/usr/lib/x86_64-linux-gnu \
  -lnvidia-ml -lmicrohttpd -ldl -pthread

Run (requires root or proper NVML permissions):

./gpu_nvml

Verify the exporter is listening:

curl http://localhost:8080/metrics

You should see Prometheus-style metrics like:

    gpu_power_usage{gpu="0"} 10680
    gpu_temperature{gpu="0"} 39
    gpu_process_memory_bytes{gpu="0",pid="1234"} 208000000

Setting Up & Running train_resnet.py

    Activate your Python virtual environment:

source venv/bin/activate

Install dependencies (if not done):

pip install torch torchvision zeus prometheus_client

Run the training＋exporter script:

python train_resnet.py

By default, it starts a Prometheus exporter on port 8000 and runs all four experiments in sequence.
To run only early stopping, you can modify main() to include just EarlyStoppingExperiment.

Verify the exporter:

curl http://localhost:8000/metrics

You should see metrics:

    ml_current_power_watt 59.6
    ml_total_energy_joule 145083.61

Prometheus Configuration

Here is an example prometheus.yml scrape config to collect both exporters:

global:
  scrape_interval: 15s

scrape_configs:
  - job_name: gpu_nvml
    static_configs:
      - targets: ['localhost:8080']

  - job_name: ml_training
    static_configs:
      - targets: ['localhost:8000']

Start Prometheus with:

prometheus --config.file=prometheus.yml

Usage Examples

    GPU Utilization Dashboard: In Grafana, plot

gpu_util_gpu{job="gpu_nvml"}

Per‐Process Memory:

gpu_process_memory_bytes{job="gpu_nvml"}

Training Power Over Time:

ml_current_power_watt{job="ml_training"}

Total Training Energy:

    ml_total_energy_joule{job="ml_training"}

For further customization—such as changing batch sizes, enabling only specific experiments, or adjusting scrape intervals—edit the Python script or Prometheus config as needed.
