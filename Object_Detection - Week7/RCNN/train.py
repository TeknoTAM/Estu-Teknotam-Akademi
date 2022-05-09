import torch
from torch.utils.data import TensorDataset, DataLoader

from utils.utils import get_json
from utils.dataset import get_data,RCNNDataset
from model import RCNN

def train_batch(inputs, model, optimizer, criterion):
    input, clss, deltas = inputs
    model.train()
    optimizer.zero_grad()
    _clss, _deltas = model(input)
    loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
    accs = clss == decode(_clss)
    loss.backward()
    optimizer.step()
    return loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sets = []
    for d in ["train", "val"]:
        sets.append(get_json("balloon/" + d))

    label2target = {'balloon':1, 'background':0}
    target2label = {t:l for l,t in label2target.items()}
    background_class = label2target['background']

    train = get_data(sets[0])
    val = get_data(sets[1])
    train_ds = RCNNDataset(*train)
    test_ds = RCNNDataset(*val)

    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=train_ds.collate_fn, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=test_ds.collate_fn, drop_last=True)

    rcnn = RCNN().to(device)
    criterion = rcnn.calc_loss
    optimizer = torch.optim.SGD(rcnn.parameters(), lr=1e-3)
    n_epochs = 5


    for epoch in range(n_epochs):

        _n = len(train_loader)
        for ix, inputs in enumerate(train_loader):
            loss, loc_loss, regr_loss, accs = train_batch(inputs, rcnn, optimizer, criterion)


    torch.save(rcnn.state_dict(),"model.pth")