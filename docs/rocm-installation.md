# Run Agents-A1.5 on AMD ROCm

This guide deploys the full multimodal Agents-A1.5 model on AMD GPUs using
vLLM or llama.cpp. A compatible AMD GPU driver and Docker are required.

Find your GPU target name before you start; the build steps below need it:

```bash
rocminfo | grep gfx
```

## Choosing a configuration

Weights range from 38 GB (Q8_0 GGUF) to over 70 GB (BF16), and not every
variant runs on every GPU, so pick a configuration before downloading. The
numbers below were measured on Agents-A1, which has the same architecture,
operators, and parameter count as Agents-A1.5.

| Setup | Weights | Measured on a 96 GB-class APU (gfx1151) |
|---|---|---|
| llama.cpp Q4_K_M + 262K context | 22 GB | works, 53 tok/s |
| llama.cpp Q8_0 + 262K context | 38 GB | works, 40 tok/s; weights are placed in system memory via GTT |
| vLLM BF16 + 262K context | 70+ GB | needs well over 130 GB of unified memory; on 96 GB the reachable context ceiling is about 100K |
| vLLM FP8 | 38 GB | not supported on Radeon APUs (no FP8 MoE backend); AMD Instinct MI GPUs only |

On AMD APUs the practical limit is total unified memory, not the BIOS VRAM
carve-out: with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (vLLM) or
plain HIP allocation (llama.cpp), weights spill into system memory
automatically. If vLLM reports `No available memory for the cache blocks` or
suggests a maximum model length, lower `--max-model-len` accordingly.

## Download the model

The examples in this guide download weights automatically from Hugging Face
on first start. The BF16 weights are over 70 GB, so the first start may take
an hour or more depending on your connection.

In regions where huggingface.co is slow or unreachable, download from
ModelScope on the host instead and mount the files into the container. This
route is fast (parallel workers), integrity-checked (sha256 against the
ModelScope manifest), and survives container removal:

```bash
pip install modelscope

# vLLM path: BF16 weights (70+ GB)
modelscope download --model InternScience/Agents-A1.5 \
  --local-dir ./Agents-A1.5 --max-workers 8

# llama.cpp path: Q8_0 GGUF (38 GB, includes the multimodal projector)
modelscope download --model InternScience/Agents-A1.5-Q8_0-GGUF \
  --local-dir ./Agents-A1.5-Q8_0-GGUF --max-workers 8
```

Then add a volume mount to the `docker run` commands in this guide and point
the server at the local path instead of the repository id:

- vLLM: add `-v $PWD/Agents-A1.5:/model:ro` and serve with `--model /model`.
- llama.cpp: add `-v $PWD/Agents-A1.5-Q8_0-GGUF:/models:ro` and start
  `llama-server` with `--model /models/Agents-A1.5-Q8_0.gguf --mmproj
  /models/Agents-A1.5-mmproj.gguf` instead of `--hf-repo`.

A Hugging Face mirror also works without pre-downloading: set
`HF_ENDPOINT=https://hf-mirror.com` in the server environment (both vLLM and
llama.cpp honor it). Note that llama.cpp's built-in downloader does not verify
file hashes; on unstable connections, verify the downloaded GGUF before
reporting load errors.

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

If a container with the same name already exists, remove it first with
`docker rm agents-a15-vllm` or add `--rm` to `docker run`.

### Start the model server

The first start downloads the model and may take some time. Expect roughly
five minutes of cold start for weight loading and memory profiling, and about
one additional minute for the first multimodal request (one-time Triton
kernel compilation); subsequent requests are fast.

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

On APUs with about 96 GB of unified memory this full BF16 setup does not fit;
lower `--max-model-len` to roughly 100000 or less, or use the llama.cpp path
with the Q8_0 or Q4_K_M GGUF (see "Choosing a configuration").

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

Agents-A1.5 is a thinking model: replies arrive in the `reasoning` field
first and `content` fills in after thinking completes, so set `max_tokens`
generously when testing. For best generation quality use
`temperature=0.85, top_p=0.95, top_k=20, presence_penalty=1.1`.

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

If that image URL is unreachable from your network, pass the same image as a
base64 data URI instead (fetch the JPEG from a mirror such as hf-mirror.com
first and replace `<BASE64>`):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "InternScience/Agents-A1.5",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64>"}}
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

If a container with the same name already exists, remove it first with
`docker rm agents-a15-llama` or add `--rm` to `docker run`.

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

If the clone fails inside the container, download the source tarball on the
host and copy it in (see "Appendix: restricted networks").

Build and install llama.cpp with HIP support:

Set `GPU_TARGETS` using the LLVM target name from the
[AMD GPU hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
or from `rocminfo | grep gfx`. The example below builds for `gfx950`,
`gfx1100`, and `gfx1151`; separate multiple targets with semicolons. Building
for your single GPU target roughly halves the HIP kernel compile time
(measured: 250 s for three targets versus 81 s for one).

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

After installation, make the ROCm libraries visible system-wide once so that
every `llama-*` tool starts without extra environment variables:

```bash
echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf && ldconfig
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

`--hf-repo` also downloads the multimodal projector automatically when the
repository provides one. On APUs, `--image-min-tokens 1024` is recommended
for grounding accuracy with this model family. If you pre-downloaded the GGUF
from ModelScope, mount it as described in "Download the model" and pass
`--model` and `--mmproj` instead of `--hf-repo`.

### Test the OpenAI-compatible API

From another terminal, either on the host or inside the container, send a
text request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "Agents-A1.5-Q8_0",
    "messages": [{"role": "user", "content": "Introduce yourself."}]
  }'
```

Agents-A1.5 is a thinking model: replies arrive in the `reasoning_content`
field first and `content` fills in after thinking completes, so set
`max_tokens` generously when testing.

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

If that image URL is unreachable from your network, pass the same image as a
base64 data URI instead, as shown in the vLLM section above.

## Appendix: restricted networks

If `docker pull` from docker.io fails, pull the same images through a registry
mirror and retag them:

```bash
docker pull docker.m.daocloud.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
docker tag  docker.m.daocloud.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 \
            rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
```

If `git clone` of llama.cpp fails inside the container, fetch the source
tarball on the host and copy it in:

```bash
curl -L -o llama.cpp-master.tar.gz \
  https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/master
docker cp llama.cpp-master.tar.gz agents-a15-llama:/root/
docker exec agents-a15-llama bash -c \
  'cd /root && tar xzf llama.cpp-master.tar.gz && mv llama.cpp-master llama.cpp'
```

For Hugging Face access set `HF_ENDPOINT=https://hf-mirror.com` in the server
environment (both vLLM and llama.cpp honor it). If you use the `hf` CLI to
pre-download, also set `HF_HUB_DISABLE_XET=1`, because the Xet transfer
protocol does not work through the mirror and fails with 401.
