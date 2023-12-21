import argparse
import torch
from copy import deepcopy
from src.config import *
from src.data import *
from src.model import *
from src.utils import *

def main(args):
    present_args(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_set = ALL_TAGS if args.exp_type == 'ALL' else SELECTED_TAGS
    num_labels = len(label_set)
    trainset = MultiNERDMultiTaskDataset(lang='en', split='train', label_set=label_set, max_len=args.max_len, model_name=MODEL_MAPPING[args.model_name])
    valset = MultiNERDMultiTaskDataset(lang='en', split='val', label_set=label_set, max_len=args.max_len, model_name=MODEL_MAPPING[args.model_name])
    trainloader, valloader = prepare_loader(trainset, args.batch_size), prepare_loader(valset, args.batch_size)
    model = init_model(args.model_name, num_labels)
    optimizer, scheduler = prepare_trainer(model, trainloader, args.num_epoch, args.lr)
    model.zero_grad()
    model.cuda() if device != 'cpu' else model
    seed_all(args.seed)
    total_loss = 0.0
    best_val_f1 = 0.0
    best_model = deepcopy(model)
    train_loss_log, val_loss_log, val_m_loss_log, val_b_loss_log, val_i_loss_log, val_o_loss_log = [], [], [], [], [], []
    val_f1_log = []
    steps = []
    print("Starting training...")
    for epoch in range(0, args.num_epoch):
        print(f"[ Epoch {epoch + 1}/{args.num_epoch} ]")
        for step, batch in tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Training Epoch {epoch + 1}"):
            model.train()
            batch = [t.to(device) for t in batch]
            batch_ids, batch_labels, batch_masks, batch_b_labels, batch_i_labels, batch_o_labels = batch
            optimizer.zero_grad()
            outputs = model(
                batch_ids, attention_mask=batch_masks,
                labels=batch_labels,
                b_labels=batch_b_labels,
                i_labels=batch_i_labels,
                o_labels=batch_o_labels)
            loss: torch.Tensor = outputs[0]
            m_loss: torch.Tensor = outputs[2][0] # main
            b_loss: torch.Tensor = outputs[2][1] # b-task
            i_loss: torch.Tensor = outputs[2][2] # i-task
            o_loss: torch.Tensor = outputs[2][3] # o-task
            cur_step = epoch * len(trainloader) + (step + 1) * args.batch_size
            total_loss += loss.item()
            if step % 100 == 0:
                avg_loss = total_loss / (cur_step)
                val_outputs = multi_task_ner_inference(model, valloader, device)
                val_preds, val_labels, val_loss = val_outputs
                val_f1 = eval(val_preds, val_labels)[0]['macro_f1']
                print(f'Step: {step + 1} / {len(trainloader)}, train-loss: {avg_loss:.4f}, val-loss: {val_loss:.4f}, val-f1: {val_f1:.4f}')
                train_loss_log.append(avg_loss)
                val_loss_log.append(val_loss)
                val_f1_log.append(val_f1)
                val_m_loss_log.append(m_loss.item())
                val_b_loss_log.append(b_loss.item())
                val_i_loss_log.append(i_loss.item())
                val_o_loss_log.append(o_loss.item())
                steps.append(cur_step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        val_outputs = multi_task_ner_inference(model, valloader, device)
        val_preds, val_labels, _ = val_outputs
        val_f1 = eval(val_preds, val_labels)[0]['macro_f1']
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = deepcopy(model)
    logs = {
        'step': steps,
        'train_loss': train_loss_log,
        'val_loss': val_loss_log,
        'val_m_loss': val_m_loss_log,
        'val_b_loss': val_b_loss_log,
        'val_i_loss': val_i_loss_log,
        'val_o_loss': val_o_loss_log,
        'val_f1': val_f1_log}
    logs = pd.DataFrame.from_dict(logs)
    best_val_f1 = f'{val_f1_log[-1]:.4f}'
    folder_path = os.path.join(RESULT, f"{args.model_name}_{args.exp_type}_{best_val_f1}")
    save_exp_results(
        logs,
        best_model,
        folder_path=folder_path,
        save_model=True)
    save_json(args.__dict__, file_path=os.path.join(folder_path, 'args.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--exp_type', type=str, default='FILTERED', choices=['ALL', 'FILTERED'])
    parser.add_argument('--model_name', type=str, default='BertNERMultiTask', choices=['BertNERMultiTask'])
    parser.add_argument('--max_len', type=int, default=42)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()
    main(args)