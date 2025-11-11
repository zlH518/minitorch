ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

source /volume/pt-train/users/zlh/slime/tool_call/conda/bin/activate vllm


# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


if [ "$RANK" = "0" ]; then
        echo "Starting Ray head node..."
        ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats
python3 -c '
import ray
import time

@ray.remote
def check_nodes():
    return True

# 确保Ray已经初始化
if not ray.is_initialized():
    ray.init(address="auto")

expected_nodes = 8
max_wait_time = 900  # 最多等待300秒
start_time = time.time()

print(f"Waiting for {expected_nodes} nodes to be ready...")
while time.time() - start_time < max_wait_time:
    nodes = ray.nodes()
    alive_nodes = sum(1 for node in nodes if node["alive"])
    print(f"Current number of nodes: {alive_nodes}/{expected_nodes}")
    
    if alive_nodes >= expected_nodes:
        print("All nodes are ready!")
        break
    
    time.sleep(5)

if time.time() - start_time >= max_wait_time:
    print("Timeout waiting for nodes to be ready")
    exit(1)
'
else
    max_attempts=60
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if ray start --address="$MASTER_ADDR":6379; then
            echo "Successfully connected to head node"
            break
        else
            echo "Attempt $attempt/$max_attempts: Connection failed, retrying in 5s..."
            sleep 5
            attempt=$((attempt + 1))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo "Failed to connect after $max_attempts attempts"
        exit 1
    fi
    sleep 999999
fi


vllm serve /volume/pt-train/models/DeepSeek-V3 \
    --served-model-name deepseek-V3 \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-backend=ray \
    --enable-auto-tool-choice \
    --chat-template /volume/pt-train/models/DeepSeek-V3/tool_chat_template_deepseekv3.jinja \
    --tool-call-parser deepseek_v3
