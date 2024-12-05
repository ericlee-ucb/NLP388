# Train - squad (70%), adversarial_qa/adversarialQA (30%)

`python3 run.py --do_train --dataset squad#0.7;adversarial_qa:adversarialQA#0.3 --per_device_train_batch_size 24 --output_dir ./case_s70_adv30/model`

# Evaluation

`python3 run.py --do_eval --dataset squad --model ./case_s70_adv30/model --output_dir ./case_s70_adv30/eval_squad`
`python3 run.py --do_eval --dataset legacy107/newsqa --model ./case_s70_adv30/model --output_dir ./case_s70_adv30/eval_squad_newsqa` 
`python3 run.py --do_eval --dataset squad_adversarial:AddSent --model ./case_s70_adv30/model --output_dir ./case_s70_adv30/eval_squadadv`
