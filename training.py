import torch
import model
import util
import json
from tensorboardX import SummaryWriter
from util.data_loader import *
import util.utils as utils
import time

def train(args, unmix, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    unmix.stft.center = True
    pbar = tqdm.tqdm(train_sampler)
    for data in pbar:
        pbar.set_description("Training batch")
        x = data[0]  # mix
        y = data[1]  # target
        z = data[2]  # text
        x, y, z = x.to(device), y.to(device), z.to(device)
        print(x.shape)
        optimizer.zero_grad()
        # if args.alignment_from:
        #     inputs = (x, z, data[3].to(device))  # add attention weights to input
        # else:
        inputs = (x, z)
        Y_hat = unmix(inputs)
        Y = unmix.transform(y)
        loss_fn = torch.nn.L1Loss(reduction='sum')
        loss = loss_fn(Y_hat, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unmix.parameters(), max_norm=2, norm_type=1)
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
    return losses.avg
def valid(args, unmix, device, valid_sampler):
    losses = utils.AverageMeter()

    unmix.eval()
    unmix.stft.center = True
    with torch.no_grad():
        for data in valid_sampler:
            x = data[0]  # mix
            y = data[1]  # vocals
            z = data[2]  # text
            x, y, z = x.to(device), y.to(device), z.to(device)
            # if args.alignment_from:
            #     inputs = (x, z, data[3].to(device))  # add attention weight to input
            # else:
            inputs = (x, z)
            Y_hat = unmix(inputs)
            Y = unmix.transform(y)
            loss_fn = torch.nn.L1Loss(reduction='sum')  # in sms project, the loss is defined before looping over epochs
            loss = loss_fn(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        return losses.avg #, sdr_avg.avg, sar_avg.avg, sir_avg.avg

def train_model(specs, model):
    model_path_name = specs["name"]
    sr = specs["sample_rate"]
    n_fft = specs["n_fft"]
    n_hop = specs["n_hop"]
    lr = specs["lr"]
    weight_decay = specs["weight_decay"]
    batch_size = specs["batch-size"]
    # random seeds
    torch.manual_seed(specs["seed"])
    random.seed(specs["seed"])
    np.random.seed(specs["seed"])

    # setting up training environment and variables
    writer = SummaryWriter(logdir=os.path.join('tensorboard', model_path_name))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using GPU:", use_cuda)
    print("Using Torchaudio: ", utils._torchaudio_available())
    t = tqdm.trange(1, specs["epochs"] + 1)
    # save path
    target_path = Path("trained_models/{}/".format(model_path_name))
    target_path.mkdir(parents=True, exist_ok=True)

    # training and validation datasets
    if specs["dataset"] == "TIMIT":
        train_dataset = TIMITMusicTrain(None, fixed_length=True, mono=True)
        valid_dataset = TIMITMusicTest(None, fixed_length=True, size=500, mono=True)
    elif specs["dataset"] == "NUS":
        train_dataset = NUSMusicTrain(None, fixed_length=True, mono=True)
        valid_dataset = NUSMusicTest(None, fixed_length=True, size=500, mono=True)
    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    # prep optimizer
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    # schedulaer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=specs["lr-decay-gamma"],
        patience=specs["lr-decay-patience"],
        cooldown=10
    )
    es = utils.EarlyStopping(patience=specs["patience"])

    if specs["pre_train_location"] != "":
        model_path = Path(os.path.join('trained_models/', specs["pre_train_location"])).expanduser()
        with open(Path(os.path.join(model_path, "vocal" + '.json')), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, "vocal" + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)


        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # train for another arg.epochs
        t = tqdm.trange(
            results['epochs_trained'],
            results['epochs_trained'] + specs["epochs"] + 1,
        )
        train_losses = results['train_loss_history']
        valid_losses = results['valid_loss_history']
        train_times = results['train_time_history']
        best_epoch = 0

    # else start from 0
    else:
        t = tqdm.trange(1, specs["epochs"] + 1)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    # training loop
    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = train(specs, model, device, train_sampler, optimizer)
        valid_loss = valid(specs, model, device, valid_sampler)
        # valid_loss = valid(args, model, device, valid_sampler)
        writer.add_scalar("Training_cost", train_loss, epoch)
        writer.add_scalar("Validation_cost", valid_loss, epoch)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )
        stop = es.step(valid_loss)
        if valid_loss == es.best:
            best_epoch = epoch
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': es.best,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        },
            is_best=valid_loss == es.best,
            path=target_path,
            target="vocal"
        )

        # save params
        params = {
            'epochs_trained': epoch,
            'args': specs,
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs
        }

        with open(Path(target_path, "vocal" + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break



if __name__ == "__main__":

    # # input dict
    # with open("training_specs/toy_example_unmix_pretrain.json") as f:
    #     specs = json.load(f)
    # # input_specs
    # model_to_train = model.OpenUnmix(sample_rate=specs["sample_rate"], n_fft=specs["n_fft"], n_hop=specs["n_hop"], input_is_spectrogram=True)
    # train_model(specs, model_to_train)

    with open("training_specs/toy_example_unmix.json") as f:
        specs = json.load(f)
    # input_specs
    model_to_train = model.OpenUnmix(sample_rate=specs["sample_rate"], n_fft=specs["n_fft"], n_hop=specs["n_hop"], input_is_spectrogram=True)
    train_model(specs, model_to_train)
