import torch
import yaml
from tqdm import tqdm
import numpy as np
from data.dataloader import load_data
from model.network import create_model
from sklearn.metrics import balanced_accuracy_score as bas
import random

# random.seed(2)
# np.random.seed(2)
# torch.manual_seed(2)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(2)

def get_train_embeddings(dataloader, model, device):
	model = model.to(device)
	gnn_embeddings, bert_embeddings = [], []
	labels = []

	for batch in tqdm(dataloader, leave=False):
		batch = batch.to(device)
		output = model(batch)

		bert_embeddings.append(output[0].detach().cpu().numpy())
		gnn_embeddings.append(output[1].detach().cpu().numpy())
		labels.extend(batch.label.cpu().numpy())

	return np.vstack(bert_embeddings).T, np.vstack(gnn_embeddings).T, labels


def main(train_data_loader, val_data_loader, model, device):
	bert_embeddings, gnn_embeddings, labels = get_train_embeddings(train_data_loader, model, device)

	bert_labels, gnn_labels = [], []
	ground_truth = []
	num_correctb, num_correctg = 0, 0
	for batch in tqdm(val_data_loader, leave=False):
		batch = batch.to(device)
		output = model(batch)

		bert_output = output[0].detach().cpu().numpy()
		gnn_output = output[1].detach().cpu().numpy()

		bert_pred = np.argmax(
			bert_output @ bert_embeddings,
			axis=1
		)
		gnn_pred = np.argmax(
			gnn_output @ gnn_embeddings,
			axis=1
		)

		bert_label = list(map(labels.__getitem__, bert_pred))
		gnn_label = list(map(labels.__getitem__, gnn_pred))

		num_correctb += np.sum(
			bert_label == batch.label.cpu().numpy()
		)
		num_correctg += np.sum(
			gnn_label == batch.label.cpu().numpy()
		)

		bert_labels.extend(bert_label)
		gnn_labels.extend(gnn_label)
		ground_truth.extend(batch.label.cpu().numpy())

	print(f'BERT Accuracy: {num_correctb / len(val_data_loader.dataset)}')


	return num_correctb / len(val_data_loader.dataset)


if __name__ == '__main__':
	device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
	print(f'Device: {device}\n')

	config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
	config['device'] = device

	train_data_loader, val_data_loader = load_data(config)

	model = create_model(config)
	model.eval()


	BERT_state_dict = torch.load(f'./checkpoints/{config["task"]}/model.pt', map_location=device)

	# BERT_state_dict = torch.load(f'./checkpoints/updated_inference/nf/model.pt', map_location=device)

	model.load_state_dict(BERT_state_dict['bert_state_dict'], strict=False)


	main(train_data_loader, val_data_loader, model, device)



	#源代码 效果差
    # BERT_state_dict = torch.load(f'./checkpoints/{config["task"]}/model.pt', map_location=device)
	# model.bert.load_state_dict(BERT_state_dict['model_state_dict'], strict=False)

#
# import torch
# import yaml
# from tqdm import tqdm
# import numpy as np
# from data.dataloader import load_data
# from model.network import create_model
# from sklearn.metrics import balanced_accuracy_score as bas
# from transformers import EsmTokenizer
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def get_train_embeddings(dataloader, model, device):
# 	model = model.to(device)
# 	gnn_embeddings, bert_embeddings = [], []
# 	labels = []
#
# 	with torch.no_grad():
# 		for batch in tqdm(dataloader, leave=False):
# 			batch = batch.to(device)
# 			output = model(batch)
#
# 			bert_embeddings.append(output[0].detach().cpu().numpy())
# 			gnn_embeddings.append(output[1].detach().cpu().numpy())
# 			labels.extend(batch.label.cpu().numpy())
#
# 	return np.vstack(bert_embeddings).T, np.vstack(gnn_embeddings).T, labels
#
#
# def main(train_data_loader, val_data_loader, model, device):
# 	bert_embeddings, gnn_embeddings, labels = get_train_embeddings(train_data_loader, model, device)
#
# 	num_correctb = 0
# 	with torch.no_grad():
# 		for batch in tqdm(val_data_loader, leave=False):
# 			batch = batch.to(device)
# 			output = model(batch)
#
# 			bert_output = output[0].detach().cpu().numpy()
# 			gnn_output = output[1].detach().cpu().numpy()
#
# 			bert_pred = np.argmax(bert_output @ bert_embeddings, axis=1)
# 			bert_label = list(map(labels.__getitem__, bert_pred))
#
# 			num_correctb += np.sum(bert_label == batch.label.cpu().numpy())
#
# 	accuracy = num_correctb / len(val_data_loader.dataset)
# 	print(f'BERT Accuracy: {accuracy:.4f}')
# 	return accuracy
#
#
# if __name__ == '__main__':
# 	# 加载配置（与main代码保持一致）
# 	config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
# 	config['device'] = device
# 	config['tokenizer'] = EsmTokenizer.from_pretrained("/mnt/sdb/cjj/MultiPeptide-main/esm-2")  # 添加tokenizer配置
#
# 	# 加载数据
# 	train_data_loader, val_data_loader = load_data(config)
#
# 	# 创建模型
# 	model = create_model(config).to(device)
# 	model.eval()
#
# 	# 加载训练好的参数（关键修改部分）
# 	checkpoint = torch.load(f'./checkpoints/{config["task"]}/model.pt', map_location=device)
#
# 	# 分别加载GNN和ESM参数
# 	model.gnn.load_state_dict(checkpoint['gnn_state_dict'])
# 	model.bert.esm.load_state_dict(checkpoint['esm_state_dict'])
#
# 	# 可选：加载优化器和调度器状态（如果需要继续训练）
# 	# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# 	# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#
# 	print("Loaded checkpoint from epoch:", checkpoint['epoch'] + 1)
# 	print(f"Training loss: {checkpoint['train_loss']:.4f}")
# 	print(f"Validation loss: {checkpoint['val_loss']:.4f}\n")
#
# 	# 运行推理
# 	main(train_data_loader, val_data_loader, model, device)