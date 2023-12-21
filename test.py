import os
import argparse
from src.config import *
from src.data import *
from src.model import *
from src.utils import *

def main(args):
    # Display arguments for logging purposes
    present_args(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint and configuration
    ckp_path = os.path.join(RESULT, args.ckp_name)
    if not os.path.exists(ckp_path):
        raise NameError(f"Checkpoint '{args.ckp_name}' not found.")
    params = load_json(os.path.join(ckp_path, 'args.json'))

    # Determine label set based on experiment type
    label_set = ALL_TAGS if params['exp_type'] == 'ALL' else SELECTED_TAGS
    id2label = {label_set[k]: k for k in label_set}
    num_labels = len(label_set)

    # Prepare test dataset and loader
    testset = MultiNERDDataset(lang='en', split='test', label_set=label_set, max_len=params['max_len'], model_name=MODEL_MAPPING[params['model_name']])
    testloader = prepare_loader(testset, args.batch_size)

    # Load model from checkpoint
    model = init_model(params['model_name'], num_labels)
    model = load_model(model, ckp_path)
    model.to(device)
    preds, labels, _ = ner_inference(model, testloader, device)
    results, preds, labels = eval(preds, labels)
    save_json(results, file_path=os.path.join(ckp_path, 'res.json'))

    # Save organized tags as JSON
    '''
    organized_tags = {}
    for idx, (pred, true) in enumerate(zip(preds, labels)):
        tokens = testset.dataset.iloc[idx]['tokens']
        pred_tags = [id2label[i] for i in pred]
        true_tags = [id2label[i] for i in true]
        organized_tags [idx] = {
            "tokens": tokens,
            "true_tag": true_tags,
            "pred_tag": pred_tags,
            "pred_idx": pred,
            "true_idx": true
        }
    with open(os.path.join(ckp_path, 'tags.json'), 'w') as f:
        json.dump(organized_tags, f, indent=4)
    '''
    organized_tags = {}
    with open(os.path.join(ckp_path, 'tags.json'), 'w') as f:
        for idx, (pred, true) in enumerate(zip(preds, labels)):
            temp = {}
            tokens = testset.dataset.iloc[idx]['tokens']
            pred_tags = [id2label[i] for i in pred]
            true_tags = [id2label[i] for i in true]
            temp['tokens'] = tokens
            temp['true_tag'] = true_tags
            temp['true_idx'] = true
            temp['pred_tag'] = pred_tags
            temp['pred_idx'] = pred
            organized_tags[idx] = temp
        json.dump(organized_tags, f, indent=4)
    print(f"Done! result is written to the json file")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, default='RoBERTaNEROriginal_FILTERED')
    parser.add_argument('--batch_size', type=int, default=48)
    args = parser.parse_args()
    main(args)