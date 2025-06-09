
# CDs_and_Vinyl
category="CDs_and_Vinyl"
tdir=test
train_file=$(ls -f data/train/${category}*11.csv)
eval_file=$(ls -f data/valid/${category}*11.csv)
test_file=$(ls -f data/${tdir}/${category}*11.csv)
info_file=$(ls -f data/info/${category}*.txt)
echo ${train_file} ${test_fie} ${info_file} ${eval_file}
port=$((RANDOM % 5000 + 10000))
export CUDA_VISIBLE_DEVICES=6,7
task_name="latent_${category}"
out=./output_dir/${category}/latent
mkdir -p ${out}

# Replace with your actual base model path
# for example: Qwen/Qwen2.5-1.5B
BASE_MODEL=/path/to/your/base/model  
torchrun --nproc_per_node=2 --nnodes=1 --master_port=$port \
    src/latent/latent_attention_train.py \
    --base_model $BASE_MODEL \
    --train_file ${train_file} --eval_file ${eval_file} \
    --output_dir ${out} --category ${category} --sample -1 \
    > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

# available gpu
array=(3 7)

# total number of gpus
nn=2
mkdir -p ./output_dir/${category}_base
python src/utils/split.py --input_path $test_file \
        --output_path ./output_dir/${category}_base --nn ${nn}

model_ckpt=./output_dir/${category}/latent
for (( i=0; i<$nn; i++ ))
do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    echo ${category}_base/${i}.csv
    python src/latent/latent_attention_eval.py --batch_size 24 \
        --base_model $model_ckpt --sample -1 \
        --info_file ${info_file} --category ${category} \
        --test_data_path ./output_dir/${category}_base/${i}.csv \
        --result_json_data ./output_dir/${category}_base/${i}.json &
done
wait
python src/utils/merge.py --input_path ./output_dir/${category}_base/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file}

