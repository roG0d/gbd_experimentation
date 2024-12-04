Run server:
```
docker run --gpus all \
    --name sglang_server \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 30000 --grammar-backend outlines
```