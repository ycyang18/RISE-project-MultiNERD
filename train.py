import torch
import argparse
from tqdm import tqdm
from copy import deepcopy
from src.config import *
from src.data import *
from src.model import *
from src.utils import *

def main(args):
    # Initialize device and model configuration
    present_args(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_set = ALL_TAGS if args.exp_type == 'ALL' else SELECTED_TAGS
    num_labels = len(label_set)
    #traindata = load_multiNERD(split='train', save=False)
    #valdata = load_multiNERD(split='val', save=False)
    print(f"Preparing dataset... ")
    trainset = MultiNERDDataset(lang='en', split='train', label_set=label_set, max_len=args.max_len, model_name=MODEL_MAPPING[args.model_name])
    valset = MultiNERDDataset(lang='en', split='val', label_set=label_set, max_len=args.max_len, model_name=MODEL_MAPPING[args.model_name])
    trainloader, valloader = prepare_loader(trainset, args.batch_size), prepare_loader(valset, args.batch_size)
    model = init_model(args.model_name, num_labels)
    optimizer, scheduler = prepare_trainer(model, trainloader, args.num_epoch, args.lr)
    model.zero_grad()
    model.to(device)
    seed_all(args.seed)
    total_loss = 0.0
    best_val_f1 = 0.0
    best_model = deepcopy(model)
    train_loss_log, val_loss_log, val_f1_log, steps = [], [], [], []
    for epoch in range(0, args.num_epoch):
        print(f"[ Epoch {epoch + 1}/{args.num_epoch} ]")
        for step, batch in tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Training Epoch {epoch + 1}"):
            model.train()
            batch_ids, batch_labels, batch_masks = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(batch_ids, attention_mask=batch_masks, labels=batch_labels)
            loss: torch.Tensor = outputs[0]
            current_step = epoch * len(trainloader) + (step + 1) * args.batch_size
            total_loss += loss.item()
            if step % 100 == 0:
                avg_loss = total_loss / (current_step)
                val_outputs = ner_inference(model, valloader, device)
                val_preds, val_labels, val_loss = val_outputs
                val_f1, val_acc = eval(val_preds, val_labels)[0]['macro_f1'], eval(val_preds, val_labels)[0]['accuracy(micro_f1)']
                print(f" -- Epoch: {epoch + 1}/{args.num_epoch}, Step: {step + 1}/{len(trainloader)} -- ")
                print(f"  Training Loss: {avg_loss:.4f}")
                print(f"  Validation Loss: {val_loss:.4f}, F1 Score: {val_f1:.4f}, Accuracy: {val_acc:.4f}")
                train_loss_log.append(avg_loss)
                val_loss_log.append(val_loss)
                val_f1_log.append(val_f1)
                steps.append(current_step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        val_outputs = ner_inference(model, valloader, device)
        val_preds, val_labels, _ = val_outputs
        val_f1, val_acc = eval(val_preds, val_labels)[0]['macro_f1'], eval(val_preds, val_labels)[0]['accuracy(micro_f1)']
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = deepcopy(model)
    logs = {
        'step': steps,
        'train_loss': train_loss_log,
        'val_loss': val_loss_log,
        'val_f1': val_f1_log}
    logs = pd.DataFrame.from_dict(logs)
    best_val_f1 = f'{val_f1_log[-1]:.4f}'
    folder_path = os.path.join(RESULT, f"{args.model_name}_{args.exp_type}")
    save_exp_results(
        logs,
        best_model,
        folder_path=folder_path,
        save_model=True)
    save_json(args.__dict__, file_path=os.path.join(folder_path, 'args.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--exp_type', type=str, default='ALL', choices=['ALL', 'FILTERED'])
    parser.add_argument('--model_name', type=str, default='BertNERLSTM', choices=ALL_MODELS.keys())
    parser.add_argument('--max_len', type=int, default=42)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()
    main(args)