# category="Toys_and_Games"

for category in "Toys_and_Games"; do

    path_to_data=data
    train_file=$(ls -f ${path_to_data}/train/${category}*11.csv)
    eval_file=$(ls -f ${path_to_data}/valid/${category}*11.csv)
    test_file=$(ls -f ${path_to_data}/test/${category}*11.csv)
    info_file=$(ls -f ${path_to_data}/info/${category}*.txt)
    for lr in 1e-4 1e-5 5e-4 5e-5; do
        export CUDA_VISIBLE_DEVICES=3,4
        task_name=nomul_attn_lr_${lr}
        base_model=output_dir/${category}/latent
        out=./output_dir/${category}/${task_name}
        mkdir -p $out
        torchrun --nnodes=1 --nproc_per_node=2 --master_port=$((RANDOM % 9999 + 10000)) \
            src/grpo_attention_tuning/train_noise_grpo.py --category $category \
            --num_generations 8 --sample -1 --num_epochs 1 --info_file $info_file \
            --base_model $base_model --output_dir $out --beta 0.00 \
            --batch_size 256 --micro_batch_size 8 --lr $lr --seed 42 \
            --train_file $train_file --eval_file $eval_file \
            --num_iterations 1 --epsilon 0.2 --epsilon_high 0.28 \
            > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

        # available gpu
        array=(3 4)
        nn=2
        model_ckpt=${out}
        mkdir -p ./output_dir/${category}_base
        python src/utils/split.py --input_path $test_file \
                --output_path ./output_dir/${category}_base --nn ${nn}

        # 遍历数组并输出内容
        for (( i=0; i<$nn; i++ ))
        do
            export CUDA_VISIBLE_DEVICES=${array[$i]}
            echo output_dir/${category}_base/${i}.csv
            python src/grpo_attention_tuning/noise_eval.py \
                    --batch_size 12 --base_model $model_ckpt --sample -1 \
                    --info_file ${info_file} --category ${category} --end_k -1 \
                    --test_data_path ./output_dir/${category}_base/${i}.csv \
                    --result_json_data ./output_dir/${category}_base/${i}.json &
            done
        wait
        python src/utils/merge.py --input_path ./output_dir/${category}_base/ \
                --output_path ${model_ckpt}/final_result.json --nn ${nn}
    done

done

