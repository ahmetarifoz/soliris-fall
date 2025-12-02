# Fall Detection Agent

Real-time fall detection pipeline that ingests RTSP streams (or any OpenCV-compatible source), tracks humans with YOLO pose inference, and emits webhook alerts once a fall is confirmed. The application can render a diagnostic HUD overlay, save annotated footage, and tolerate temporary RTSP drops via automatic reconnection.

## Features
- **YOLO pose tracking** with temporal heuristics to distinguish falls from normal motion.
- **RTSP-friendly runtime** including configurable FFMPEG options, frame dropping, and reconnect logic to keep latency low.
- **Structured alerts** delivered over HTTP with signed payloads and explicit fall/candidate status.
- **Custom HUD overlay** summarising per-track metrics in real time.
- **Configurable thresholds** via YAML (sample provided in ).
- **API-controlled detection** - Detection starts disabled by default, enable via REST API (`POST /enable`).
- **Lazy stream initialization** - Stream reading only starts when detection is enabled, minimizing resource usage.

## Getting Started
1. Install Python 3.10+ and create a virtual environment:
   
2. Install dependencies:
   Collecting pip
  Using cached pip-25.2-py3-none-any.whl (1.8 MB)
Installing collected packages: pip
Successfully installed pip-25.2
Collecting ultralytics==8.3.197
  Using cached ultralytics-8.3.197-py3-none-any.whl (1.1 MB)
Collecting opencv-python>=4.10.0
  Using cached opencv_python-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (67.0 MB)
Collecting numpy>=1.24.0
  Using cached numpy-2.0.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (19.5 MB)
Collecting lap>=0.5.12
  Using cached lap-0.5.12-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.7 MB)
Collecting torch>=2.2.0
  Downloading torch-2.6.0-cp39-cp39-manylinux1_x86_64.whl (766.7 MB)
Requirement already satisfied: pyyaml>=6.0.0 in /home/ahmetarifoz/.local/lib/python3.9/site-packages (from -r requirements.txt (line 6)) (6.0.3)
Collecting requests>=2.31.0
  Using cached requests-2.32.5-py3-none-any.whl (64 kB)
Collecting scipy>=1.4.1
  Downloading scipy-1.13.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (38.6 MB)
Collecting matplotlib>=3.3.0
  Using cached matplotlib-3.9.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.3 MB)
Collecting ultralytics-thop>=2.0.0
  Downloading ultralytics_thop-2.0.17-py3-none-any.whl (28 kB)
Collecting pillow>=7.1.2
  Using cached pillow-11.3.0-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (7.6 MB)
Collecting torchvision>=0.9.0
  Using cached torchvision-0.21.0-cp39-cp39-manylinux1_x86_64.whl (7.2 MB)
Collecting psutil
  Using cached psutil-7.1.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (291 kB)
Collecting polars
  Downloading polars-1.33.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (39.7 MB)
Collecting sympy==1.13.1; python_version >= "3.9"
  Downloading sympy-1.13.1-py3-none-any.whl (6.2 MB)
Collecting nvidia-cuda-runtime-cu12==12.4.127; platform_system == "Linux" and platform_machine == "x86_64"
  Downloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (883 kB)
Requirement already satisfied: filelock in /home/ahmetarifoz/.local/lib/python3.9/site-packages (from torch>=2.2.0->-r requirements.txt (line 5)) (3.16.1)
Collecting nvidia-cufft-cu12==11.2.1.3; platform_system == "Linux" and platform_machine == "x86_64"
  Downloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)
Collecting nvidia-curand-cu12==10.3.5.147; platform_system == "Linux" and platform_machine == "x86_64"
  Downloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)
Collecting nvidia-cusolver-cu12==11.6.1.9; platform_system == "Linux" and platform_machine == "x86_64"
  Downloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)
Collecting nvidia-nvtx-cu12==12.4.127; platform_system == "Linux" and platform_machine == "x86_64"
  Using cached nvidia_nvtx_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (99 kB)
Collecting triton==3.2.0; platform_system == "Linux" and platform_machine == "x86_64"
  Downloading triton-3.2.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (253.1 MB)
Collecting fsspec
  Using cached fsspec-2025.9.0-py3-none-any.whl (199 kB)
Collecting networkx
  Downloading networkx-3.2.1-py3-none-any.whl (1.6 MB)
Collecting nvidia-cuda-cupti-cu12==12.4.127; platform_system == "Linux" and platform_machine == "x86_64"
  Downloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (13.8 MB)
Requirement already satisfied: typing-extensions>=4.10.0 in /home/ahmetarifoz/.local/lib/python3.9/site-packages (from torch>=2.2.0->-r requirements.txt (line 5)) (4.15.0)
Collecting nvidia-nvjitlink-cu12==12.4.127; platform_system == "Linux" and platform_machine == "x86_64"
  Using cached nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
Collecting nvidia-cudnn-cu12==9.1.0.70; platform_system == "Linux" and platform_machine == "x86_64"
  Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)
Collecting nvidia-cusparselt-cu12==0.6.2; platform_system == "Linux" and platform_machine == "x86_64"
  Downloading nvidia_cusparselt_cu12-0.6.2-py3-none-manylinux2014_x86_64.whl (150.1 MB)
Collecting nvidia-cusparse-cu12==12.3.1.170; platform_system == "Linux" and platform_machine == "x86_64"
  Using cached nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)
Requirement already satisfied: jinja2 in /usr/lib/python3/dist-packages (from torch>=2.2.0->-r requirements.txt (line 5)) (2.10.1)
Collecting nvidia-cuda-nvrtc-cu12==12.4.127; platform_system == "Linux" and platform_machine == "x86_64"
  Downloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (24.6 MB)
Collecting nvidia-nccl-cu12==2.21.5; platform_system == "Linux" and platform_machine == "x86_64"
  Using cached nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl (188.7 MB)
Collecting nvidia-cublas-cu12==12.4.5.8; platform_system == "Linux" and platform_machine == "x86_64"
  Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)
Requirement already satisfied (use --upgrade to upgrade): nvidia-cublas-cu12==12.4.5.8; platform_system == "Linux" and platform_machine == "x86_64" from https://files.pythonhosted.org/packages/ae/71/1c91302526c45ab494c23f61c7a84aa568b8c1f9d196efa5993957faf906/nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl#sha256=2fc8da60df463fdefa81e323eef2e36489e1c94335b5358bcb38360adf75ac9b in /home/ahmetarifoz/.local/lib/python3.9/site-packages (from torch>=2.2.0->-r requirements.txt (line 5))
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/lib/python3/dist-packages (from requests>=2.31.0->-r requirements.txt (line 7)) (1.25.8)
Requirement already satisfied: idna<4,>=2.5 in /usr/lib/python3/dist-packages (from requests>=2.31.0->-r requirements.txt (line 7)) (2.8)
Requirement already satisfied: certifi>=2017.4.17 in /usr/lib/python3/dist-packages (from requests>=2.31.0->-r requirements.txt (line 7)) (2019.11.28)
Requirement already satisfied: charset_normalizer<4,>=2 in /home/ahmetarifoz/.local/lib/python3.9/site-packages (from requests>=2.31.0->-r requirements.txt (line 7)) (3.4.3)
Requirement already satisfied: contourpy>=1.0.1 in /home/ahmetarifoz/.local/lib/python3.9/site-packages (from matplotlib>=3.3.0->ultralytics==8.3.197->-r requirements.txt (line 1)) (1.3.0)
Requirement already satisfied: fonttools>=4.22.0 in /home/ahmetarifoz/.local/lib/python3.9/site-packages (from matplotlib>=3.3.0->ultralytics==8.3.197->-r requirements.txt (line 1)) (4.60.1)
3. Copy the example configuration and edit to match your environment:
   
   - Point  to your YOLO weights under .
   - Update  with your RTSP URL or other video source.
   - Adjust detection thresholds or webhook settings as needed.
4. Place your YOLO pose weights (e.g., , , ) inside .
5. Run the detector:
   

## Alert Payload
Each confirmed event generates a JSON payload similar to:

 is  when a person has been in the candidate state beyond the configured deadline, otherwise  indicates a confirmed fall.

## Repository Layout


## RTSP Tuning
Low-latency RTSP performance depends on the camera and network. Key knobs in :
- : switch between  and  depending on packet loss vs. latency trade-offs.
- : forwarded to FFMPEG; , , and small  help minimise buffering.
- : number of frames to grab-and-discard on each read to flush stale buffers.

## Development Tips
- Use  for secrets; it will be automatically merged and is ignored by git.
- Headless deployments can disable the HUD by setting .
- Consider adding unit tests around the state machine before large refactors.

## License
Choose and add a license (e.g., MIT) before publishing the repository publicly.
