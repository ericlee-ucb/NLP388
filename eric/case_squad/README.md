# Train - squad

`python3 run.py --do_train --dataset squad --per_device_train_batch_size 24 --output_dir ./case_squad/model`


# Evaluation

`python3 run.py --do_eval --dataset squad --model ./case_squad/model --output_dir ./case_squad/eval_squad`
`python3 run.py --do_eval --dataset legacy107/newsqa --model ./case_squad/model --output_dir ./case_squad/eval_squad_newsqa`
`python3 run.py --do_eval --dataset squad_adversarial/AddSent --model ./case_squad/model --output_dir ./case_squad/eval_squadadv`
