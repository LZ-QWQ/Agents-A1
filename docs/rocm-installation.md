# Run Agents-A1.5 on AMD ROCm

This guide deploys the full multimodal Agents-A1.5 model on AMD GPUs using vLLM
or llama.cpp. A compatible AMD GPU driver and Docker are required.

## vLLM

### Pull the image

```bash
docker pull rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
```

### Start and enter the container

Start the container:

```bash
docker run --detach --name agents-a15-vllm \
  --device /dev/kfd \
  --device /dev/dri \
  --shm-size 64g \
  --publish 127.0.0.1:8000:8000 \
  --security-opt seccomp=unconfined \
  --entrypoint sleep \
  rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 \
  infinity
```

Enter the container:

```bash
docker exec -it agents-a15-vllm bash
```

### Start the model server

The first start downloads the model and may take some time.

Choose one of the following commands for your hardware.
If you are unsure which category applies to your GPU, see the
[AMD GPU hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html).

#### AMD Radeon GPUs

The BF16 weights are over 70 GB and do not fit on one Radeon GPU, so this
example uses four GPUs. Enable Triton FlashAttention for the multimodal vision
encoder:

```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
vllm serve InternScience/Agents-A1.5 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

You can use `HIP_VISIBLE_DEVICES` to select the four GPUs for this command, for
example `HIP_VISIBLE_DEVICES=0,1,2,3`.

#### AMD Ryzen APUs

Expandable memory segments and a lower sequence limit help avoid out-of-memory
errors. Enable Triton FlashAttention for the multimodal vision encoder:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
vllm serve InternScience/Agents-A1.5 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

#### AMD Instinct MI GPUs

Enable [AITER](https://github.com/ROCm/aiter)-optimized ROCm kernels:

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve InternScience/Agents-A1.5 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### Test the OpenAI-compatible API

From another terminal, either on the host or inside the container, send a text
request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "InternScience/Agents-A1.5",
    "messages": [{"role": "user", "content": "Introduce yourself."}]
  }'
```
Send a multimodal request using a public image from the Hugging Face
documentation dataset:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "InternScience/Agents-A1.5",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"}}
      ]
    }]
  }'
```

## llama.cpp

### Pull the image

```bash
docker pull rocm/dev-ubuntu-24.04:10.0.0-full
```

### Start and enter the container

Start the container:

```bash
docker run --detach --name agents-a15-llama \
  --device /dev/kfd \
  --device /dev/dri \
  --publish 127.0.0.1:8080:8080 \
  --security-opt seccomp=unconfined \
  --entrypoint sleep \
  rocm/dev-ubuntu-24.04:10.0.0-full \
  infinity
```

Enter the container:

```bash
docker exec -it agents-a15-llama bash
```

### Install llama.cpp

Install the build dependencies inside the container:

```bash
apt-get update
apt-get install -y --no-install-recommends build-essential ca-certificates cmake git libcurl4-openssl-dev libssl-dev
```

Download the source:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Build and install llama.cpp with HIP support:

Set `GPU_TARGETS` using the LLVM target name from the
[AMD GPU hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html).
The example below builds for `gfx950`, `gfx1100`, and `gfx1151`; separate
multiple targets with semicolons.

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx950;gfx1100;gfx1151" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
cmake --install build
ldconfig
```

### Start the model server

The first start downloads the model and its multimodal projector and may take
some time.

Start the server inside the container. `HIP_VISIBLE_DEVICES=0` selects GPU 0
for this single-GPU example:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib \
HIP_VISIBLE_DEVICES=0 \
llama-server \
  --hf-repo InternScience/Agents-A1.5-Q8_0-GGUF \
  --alias Agents-A1.5-Q8_0 \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 262144 \
  --n-gpu-layers all
```

### Test the OpenAI-compatible API

From another terminal, either on the host or inside the container, send a text
request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "Agents-A1.5-Q8_0",
    "messages": [{"role": "user", "content": "Introduce yourself."}]
  }'
```

Send a multimodal request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "Agents-A1.5-Q8_0",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"}}
      ]
    }]
  }'
```
