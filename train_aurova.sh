function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)

iterations=40_000
gpu=-1
ratio=1
resolution=-1
appearance_dim=0

fork=2
base_layer=12
visible_threshold=0.9
dist2level="round"
update_ratio=0.2

dist_ratio=0.999 #0.99
levels=-1
init_level=-1
extra_ratio=0.25
extra_up=0.01

time=$(date "+%y_%m_%d_%H_%M")

python train.py --eval -s /workspace/data/${NAME} -r ${resolution} --gpu ${gpu} --fork ${fork} --ratio ${ratio} \
    --iterations ${iterations} --port $port -m /workspace/outputs/${NAME}/Octotree-GS/$time --appearance_dim ${appearance_dim} \
    --visible_threshold ${visible_threshold}  --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
    --progressive --init_level ${init_level} --dist_ratio ${dist_ratio} --levels ${levels}  \
    --extra_ratio ${extra_ratio} --extra_up ${extra_up}  

