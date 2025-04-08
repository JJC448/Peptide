import torch
import wandb
from datetime import datetime
import yaml
import os
import shutil
from data.dataloader import load_data
from model.network import create_model, cri_opt_sch
from model.utils import train, validate

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')

    for epoch in range(config['epochs']):
        train_loss = train(model, train_data_loader, optimizer, criterion, scheduler, device)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {train_loss:.4f} \tLR: {curr_lr}')
        val_loss = validate(model, val_data_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Validation Loss: {val_loss:.4f}\n')
        scheduler.step(val_loss)
        if not config['debug']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': curr_lr,
                'gnn_params': sum(p.numel() for p in model.gnn.parameters()),
                'transformer_params': sum(p.numel() for p in model.gnn.transformer1.parameters())
            })

        if True:
            torch.save({
                'epoch': epoch,
                'gnn_state_dict': model.gnn.state_dict(),
                'bert_state_dict': model.bert.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': curr_lr
            }, f'./checkpoints/{config["task"]}/model.pt')
            print('Model Saved\n')

    return 'Training completed'


config = yaml.load(open('/mnt/sdb/cjj/MultiPeptide-main/config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

train_data_loader, val_data_loader = load_data(config)
config['sch']['steps'] = len(train_data_loader)

model = create_model(config)
criterion, optimizer, scheduler = cri_opt_sch(config, model)

#
# GNN_state_dict = torch.load(f'./checkpoints/individual_pretrained/balanced_data/GNN/{config["task"]}/model.pt')
# gnn_state_dict = GNN_state_dict['gnn_state_dict']

# # 调整键名并过滤 SAGEConv 层参数
# sageconv_weights = {
#     k.replace('GConv.', ''): v
#     for k, v in gnn_state_dict.items()
#     if 'GConv.lin_' in k
# }
# # 加载到当前模型的 SAGEConv 层（gnn1）
# model.gnn.gnn1.load_state_dict(sageconv_weights, strict=True)
#
# GNN_state_dict = torch.load(f'./checkpoints/individual_pretrained/balanced_data/GNN/{config["task"]}/model.pt')
# model.gnn.load_state_dict(GNN_state_dict['gnn_state_dict'])


BERT_state_dict = torch.load(f'./checkpoints/individual_pretrained/balanced_data/BERT/{config["task"]}/model.pt')
model.bert.protbert.load_state_dict(BERT_state_dict['bert_state_dict'], strict=False)


save_dir = './checkpoints/temp'
shutil.copy('/mnt/sdb/cjj/MultiPeptide-main/config.yaml', f'{save_dir}/config.yaml')
shutil.copy('/mnt/sdb/cjj/MultiPeptide-main/model/network.py', f'{save_dir}/network.py')
if not config['debug']:
    run_name = f'c{datetime.now().strftime("%m%d_%H%M")}'
    wandb.init(project='PeptideFold', name=run_name)

    save_dir = f'./checkpoints/CLIP_balanced_data/{config["task"]}'
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy('/mnt/sdb/cjj/MultiPeptide-main/config.yaml', f'{save_dir}/config.yaml')
    shutil.copy('/mnt/sdb/cjj/MultiPeptide-main/model/network.py', f'{save_dir}/network.py')

train_model()
wandb.finish()




#预训练模型修改
# import torch
# import wandb
# from datetime import datetime
# import yaml
# import os
# import shutil
# from data.dataloader import load_data
# from model.network import create_model, cri_opt_sch
# from model.utils import train, validate
# from transformers import EsmTokenizer
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def train_model():
#     print(f'{"=" * 30}{"TRAINING":^20}{"=" * 30}')
#
#     for epoch in range(config['epochs']):
#         train_loss = train(model, train_data_loader, optimizer, criterion, scheduler, device)
#         curr_lr = optimizer.param_groups[0]['lr']
#         print(f'Epoch {epoch + 1}/{config["epochs"]} - Train Loss: {train_loss:.4f} \tLR: {curr_lr}')
#         val_loss = validate(model, val_data_loader, criterion, device)
#         print(f'Epoch {epoch + 1}/{config["epochs"]} - Validation Loss: {val_loss:.4f}\n')
#         scheduler.step(val_loss)
#
#         if not config['debug']:
#             wandb.log({
#                 'train_loss': train_loss,
#                 'val_loss': val_loss,
#                 'lr': curr_lr,
#                 'gnn_params': sum(p.numel() for p in model.gnn.parameters()),
#                 'esm_params': sum(p.numel() for p in model.bert.esm.parameters())
#             })
#
#         torch.save({
#             'epoch': epoch,
#             'gnn_state_dict': model.gnn.state_dict(),
#             'esm_state_dict': model.bert.esm.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'train_loss': train_loss,
#             'val_loss': val_loss,
#             'lr': curr_lr
#         }, f'./checkpoints/{config["task"]}/model.pt')
#         print('Model Saved\n')
#
#     return 'Training completed'
#
#
# # 主程序初始化
# config = yaml.load(open('/mnt/sdb/cjj/MultiPeptide-main/config.yaml', 'r'), Loader=yaml.FullLoader)
# config['device'] = device
# config['tokenizer'] = EsmTokenizer.from_pretrained("/mnt/sdb/cjj/MultiPeptide-main/esm-2")
#
# # 准备数据
# train_data_loader, val_data_loader = load_data(config)
# config['sch']['steps'] = len(train_data_loader)
#
# # 初始化模型
# model = create_model(config)
# criterion, optimizer, scheduler = cri_opt_sch(config, model)
#
# # 准备检查点目录
# save_dir = './checkpoints/temp'
# os.makedirs(save_dir, exist_ok=True)
# shutil.copy('/mnt/sdb/cjj/MultiPeptide-main/config.yaml', f'{save_dir}/config.yaml')
# shutil.copy('/mnt/sdb/cjj/MultiPeptide-main/model/network.py', f'{save_dir}/network.py')
#
# # 初始化WandB
# if not config['debug']:
#     run_name = f'c{datetime.now().strftime("%m%d_%H%M")}'
#     wandb.init(project='PeptideFold', name=run_name)
#
#     save_dir = f'./checkpoints/CLIP_balanced_data/{config["task"]}'
#     os.makedirs(save_dir, exist_ok=True)
#     shutil.copy('/mnt/sdb/cjj/MultiPeptide-main/config.yaml', f'{save_dir}/config.yaml')
#     shutil.copy('/mnt/sdb/cjj/MultiPeptide-main/model/network.py', f'{save_dir}/network.py')
#
# # 启动训练
# train_model()
# if not config['debug']:
#     wandb.finish()