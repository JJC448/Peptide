import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, scheduler, device):

    model.train()
    bar = tqdm(dataloader, desc='Train', leave=False, dynamic_ncols=True)
    total_loss = 0.0

    for i, batch in enumerate(bar):
        batch = batch.to(device)

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(*output, batch.label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        bar.set_postfix(
            l=f'{total_loss / (i + 1):.3f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
        )
        bar.update()
    bar.close()

    total_loss = total_loss / len(dataloader)
    return total_loss


def validate(model, dataloader, criterion, device):

    model.eval()
    bar = tqdm(dataloader, desc='Val', leave=False, dynamic_ncols=True)
    total_loss = 0.0

    for i, batch in enumerate(bar):
        batch = batch.to(device)

        with torch.inference_mode():
            output = model(batch)
            loss = criterion(*output, batch.label)
            total_loss += loss.item()

        bar.set_postfix(
            l=f'{total_loss / (i + 1):.3f}'
        )
        bar.update()
    bar.close()

    total_loss = total_loss / len(dataloader)
    return total_loss



