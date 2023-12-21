import os
import argparse
from mt_exp import multi_task_ner_inference

from src.config import *
from src.data import *
from src.model import *
from src.utils import *


def main(args):
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

    testset = MultiNERDMultiTaskDataset(lang='en', split='test', label_set=label_set, max_len=params['max_len'], model_name=MODEL_MAPPING[params['model_name']])
    testloader = prepare_loader(testset, args.batch_size)
    model = init_model(params['model_name'], num_labels)
    model = load_model(model, ckp_path)
    model.cuda() if device != 'cpu' else model
    outputs = multi_task_ner_inference(model, testloader, device)
    preds, labels, _ = outputs
    results, preds, labels = eval(preds, labels)
    save_json(results, file_path=os.path.join(ckp_path, 'res.json'))

    preds = [[int(i) for i in pred] for pred in preds]
    pred_tags = [[id2label[i] for i in pred] for pred in preds]
    true_tags = [[id2label[i] for i in true] for true in labels]
    tags = {
        'pred_tag': pred_tags,
        'true_tag': true_tags,
        'pred_idx': preds,
        'true_idx': labels
    }
    tags = pd.DataFrame.from_dict(tags)
    tags.to_csv(os.path.join(ckp_path, 'tags.tsv'), index=None, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, default='BertNERMultiTask_FILTERED_0.8982')
    parser.add_argument('--batch_size', type=int, default=48)
    args = parser.parse_args()
    main(args)