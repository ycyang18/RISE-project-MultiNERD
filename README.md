# MultiNERD project

This project is the realization of the Name Entity Recognition using the [MultiNERD](https://huggingface.co/datasets/Babelscape/multinerd?row=17) dataset.

## Installation 

Clone the repository and install the dependencies:
```
git clone https://github.com/ycyang18/RISE-project-MultiNERD.git
cd RISE-project-MultiNERD
pip install -r requirements.txt 
```

## Training

```
usage: train.py train [--seed SEED] [--exp_type {ALL,FILTERED}]
                      [--model_name MODEL_NAME] [--max_len MAX_LEN]
                      [--num_epoch NUM_EPOCH] [--batch_size BATCH_SIZE]
                      [--lr LR]

arguments:
  --seed SEED         Set the random seed for reproducibility (default: 42).
  --exp_type          Specify the experiment type: ALL or FILTERED (default: ALL).
  --model_name        Choose the model type. Options are 'BertNEROriginal', 'BertNERLSTM', 
                      'RoBERTaNEROriginal' (default: BertNERLSTM).
                        
  --max_len           Maximum length for tokenization (default: 42).
  --num_epoch         Number of epochs for training (default: 3).
  --batch_size        Batch size for training (default: 32).
  --lr LR             Learning rate for the optimizer (default: 2e-5).
```

You could use the nohup command to run the training in the background by:

```
nohup python train.py > nohup/train/output.out &
```

Upon completion of the training, the following artifacts are stored in the `result/MODEL_NAME` directory:

1. `args.json`: Contains the configuration and arguments used during the training session, which will be loaded to do the testing
2. `logs.tsv`: Records the training logs and tracks the progression of key metrics at various stages
3. `model.pkl`: Model checkpoint
   

## Testing

```
usage: test.py test [--ckp_name CKP_NAME] [--batch_size BATCH_SIZE]

arguments:
  --ckp_name        Name of the checkpoint to use for testing (default: BertNERLSTM_FILTERED).
  --batch_size      Batch size for testing (default: 48).
```
Upon completion of the training, the following artifacts are stored in the result/MODEL_NAME directory:

1. `res.json`: Stores performance metrics evaluated on the testing data
2. `tags.json`: Stores side-by-side comparison of the model's predicted tags against the ground truth tags
