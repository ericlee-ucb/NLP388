import datasets
from datasets import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    def dataset_transform(name, dataset):
        if name == 'legacy107/newsqa':
            return dataset.map(lambda row: {
                'id': row['key'],
                'title': str(row['document_id']),
                'context': row['context'],
                'question': row['question'],
                'answers': {'text': row['answers'], 'answer_start': row['labels'][0]['start']}},
                remove_columns = ['document_id', 'labels', 'key']
            )
        else:
            return dataset

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    else:
        if training_args.do_eval:
            base = datasets.load_dataset(('squad'))
            dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else ('squad')
            dataset = datasets.load_dataset(*dataset_id)
            dataset['validation'] = dataset_transform(dataset_id[0], dataset['validation'])
            eval_split = 'validation'

        else:
            if args.dataset is None:
                dataset_id = ('squad')
            elif ';' in args.dataset:
                ds_set = []
                probs = []
                for ds in args.dataset.split(';'):
                    opts = ds.split('#')
                    dataset_id = tuple(opts[0].split(':'))
                    ds_set.append(datasets.load_dataset(*dataset_id))
                    probs.append(float(opts[1]))
                dataset = datasets.DatasetDict({
                    'train': interleave_datasets([ds['train'] for ds in ds_set], probabilities=probs, seed=42, stopping_strategy='all_exhausted'),
                    'validation': interleave_datasets([ds['validation'] for ds in ds_set], probabilities=probs, seed=42, stopping_strategy='all_exhausted')
                })
            else:
                dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else ('squad')
                dataset = datasets.load_dataset(*dataset_id)            

    task_kwargs = {}

    # Here we select the right model fine-tuning head
    model_class = AutoModelForQuestionAnswering

    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
    prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']

        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]

        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None

    # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
    # to enable the question-answering specific evaluation metrics
    trainer_class = QuestionAnsweringTrainer
    eval_kwargs['eval_examples'] = eval_dataset

    # Metrics
    metric_squad = evaluate.load('squad')
    #metric_squad2 = evaluate.load('squad_v2')
    metric_rouge = evaluate.load('rouge')
    metric_meteor = evaluate.load('meteor')
    metric_bleu = evaluate.load('google_bleu')
    metric_bert = evaluate.load('bertscore')
        
    def compute_metrics(preds, labels, types = None):
        combo_metrics = {}

        if types == None or metric_squad.name in types:
            metrics = metric_squad.compute(predictions=preds, references=labels)
            for key in list(metrics.keys()):
                combo_metrics[f"squad_{key}"] = metrics.pop(key)

        #metrics = metric_squad2.compute(predictions=preds, references=labels)
        #for key in list(metrics.keys()):
        #    combo_metrics[f"squad2_{key}"] = metrics.pop(key)

        predictions = [pred['prediction_text'] for pred in preds]
        references = [label['answers']['text'] for label in labels]

        if types == None or metric_rouge.name in types:        
            metrics = metric_rouge.compute(predictions=predictions, references=references)            
            for key in list(metrics.keys()):
                combo_metrics[f"rouge_{key}"] = np.mean(metrics[key])

        if types == None or metric_meteor.name in types:
            metrics = metric_meteor.compute(predictions=predictions, references=references)            
            for key in list(metrics.keys()):
                combo_metrics[f"meteor_{key}"] = np.mean(metrics[key])

        if types == None or metric_bleu.name in types:
            metrics = metric_bleu.compute(predictions=predictions, references=references)            
            for key in list(metrics.keys()):
                combo_metrics[f"bleu_{key}"] = np.mean(metrics[key])

        if types == None or metric_bert.name in types:
            metrics = metric_bert.compute(predictions=predictions, references=references, lang='en')
            for key in list(metrics.keys()):
                if isinstance(metrics[key], list):
                    combo_metrics[f"bert_{key}"] = np.mean(metrics[key])
                else:
                    combo_metrics[f"bert_{key}"] = metrics[key]
        
        return combo_metrics

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds.predictions, eval_preds.label_ids)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        predictions_by_id = {pred['id']: pred for pred in eval_predictions.predictions}
        labels_by_id = {pred['id']: pred for pred in eval_predictions.label_ids}

        keys = None
        results = []

        for _, example in enumerate(tqdm(eval_dataset)):
            id = example['id']
            pred = predictions_by_id[id]
            label = labels_by_id[id]

            example_with_prediction = dict(example)                
            example_with_prediction['predicted_answer'] = pred['prediction_text']

            if example_with_prediction['predicted_answer'] in example_with_prediction['answers']['text']:
                continue
                
            metrics = compute_metrics([pred], [label], ['squad', 'bert_score'])
            if keys == None:
                keys = list(sorted(metrics.keys()))                

            results.append([id, example['title'], example['context'], example['question'], pred['prediction_text'], '|'.join(example['answers']['text'])] + [metrics[key] for key in keys])

        df = pd.DataFrame(results, columns=['id', 'title', 'context', 'question', 'prediction', 'answers'] + keys)
        df.to_csv(os.path.join(training_args.output_dir, 'eval_predictions.csv'), header=True, index=False)


if __name__ == "__main__":
    main()