# Train - squad

`python3 run.py --do_train --dataset squad --per_device_train_batch_size 24 --output_dir ./case_squad/model`
`python3 run.py --do_train --dataset squad --max_length 256 --per_device_train_batch_size 24 --output_dir ./case_squad/model256`
`python3 run.py --do_train --dataset squad --max_length 512 --per_device_train_batch_size 24 --output_dir ./case_squad/model512`


# Evaluation

`python3 run.py --do_eval --dataset squad --model ./case_squad/model --output_dir ./case_squad/eval_squad`
`python3 run.py --do_eval --dataset legacy107/newsqa --model ./case_squad/model --output_dir ./case_squad/eval_squad_newsqa`
`python3 run.py --do_eval --dataset squad_adversarial:AddSent --model ./case_squad/model --output_dir ./case_squad/eval_squadadv`


`python3 run.py --do_eval --dataset squad --model ./case_squad/model256 --output_dir ./case_squad/eval_squad_256`
`python3 run.py --do_eval --dataset legacy107/newsqa --model ./case_squad/model256 --output_dir ./case_squad/eval_newsqa_256`
`python3 run.py --do_eval --dataset squad_adversarial:AddSent --model ./case_squad/model256 --output_dir ./case_squad/eval_squadadv_256`

`python3 run.py --do_eval --dataset squad --model ./case_squad/model512 --output_dir ./case_squad/eval_squad_512`
`python3 run.py --do_eval --dataset legacy107/newsqa --model ./case_squad/model512 --output_dir ./case_squad/eval_newsqa_512`
`python3 run.py --do_eval --dataset squad_adversarial:AddSent --model ./case_squad/model512 --output_dir ./case_squad/eval_squadadv_512`
