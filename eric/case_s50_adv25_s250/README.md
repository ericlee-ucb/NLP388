# Train - squad (50%), adversarial_qa/adversarialQA (25%), squad_v2 (25%)

`python3 run.py --do_train --dataset squad#0.5;adversarial_qa:adversarialQA#0.25;squad_v2#0.25 --per_device_train_batch_size 24 --output_dir ./case_s50_adv25_s250/model`


# Evaluation

`python3 run.py --do_eval --dataset squad --model ./case_s50_adv25_s250/model --output_dir ./case_s50_adv25_s250/eval_squad`
`python3 run.py --do_eval --dataset legacy107/newsqa --model ./case_s50_adv25_s250/model --output_dir ./case_s50_adv25_s250/eval_squad_newsqa` 
`python3 run.py --do_eval --dataset squad_adversarial:AddSent --model ./case_s50_adv25_s250/model --output_dir ./case_s50_adv25_s250/eval_squadadv`
