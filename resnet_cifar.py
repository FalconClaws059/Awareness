from torchvision.datasets import CIFAR100
from torchvision.models import resnet101
from torchvision.transforms import ToTensor, Compose, Normalize, RandomCrop, RandomHorizontalFlip
import torch
import argparse
import wandb
from utils import (EarlyStopping, GamblersLoss, compute_dataset_confidences_predictions, evaluate_coverage,
                   evaluate_with_threshold, evaluate_calibration)
from tqdm import tqdm
from create_iterable import cartesian_product_itertools
from resnet import resnet110

train_batch_size = [128]
eval_batch_size = [256]
metrics_epoch_frequency = [1]
epochs = [120]
lr = [1e-5]
weight_decay = [1e-5]
patience = [0]
eval_confidence_threshold = [0]
coverage = [0]
pretrain_epochs = [30]
method = ['gamblers']
gamblers_temperature = [70]
use_test = [False]
load_model = False
save_model = False
model_to_load = ""

combos = cartesian_product_itertools(train_batch_size, eval_batch_size, metrics_epoch_frequency, epochs,lr,weight_decay,patience, eval_confidence_threshold, coverage, pretrain_epochs, method, gamblers_temperature, use_test)

for combo in combos: 
        # Define custom default variables
    custom_defaults = {
        'train_batch_size': combo[0],
        'eval_batch_size': combo[1], 
        'metrics_epoch_frequency': combo[2], 
        'epochs': combo[3],
        'lr': combo[4],
        'weight_decay': combo[5],
        'patience': combo[6], 
        'eval_confidence_threshold': [0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.10, 0.0], 
        'coverage': [100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.], 
        'pretrain_epochs': combo[9], 
        'method': combo[10], 
        'gamblers_temperature': combo[11], 
        'use_test': combo[12]
    }

    # generate code to transform the parameters above in input argument with argparse library
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=custom_defaults['train_batch_size'])
    parser.add_argument('--eval_batch_size', type=int, default=custom_defaults['eval_batch_size'])
    parser.add_argument('--metrics_epoch_frequency', type=int, default=custom_defaults['metrics_epoch_frequency'])
    parser.add_argument('--epochs', type=int, default=custom_defaults['epochs'])
    parser.add_argument('--lr', type=float, default=custom_defaults['lr'])
    parser.add_argument('--weight_decay', type=float, default=custom_defaults['weight_decay'])

    parser.add_argument('--patience', type=int, default=custom_defaults['patience'])

    parser.add_argument('--eval_confidence_threshold', type=float, nargs='+',
                        default=custom_defaults['eval_confidence_threshold'],
                        help='use during evaluation to filter out examples where the model is not confident enough')

    parser.add_argument('--coverage', type=float, nargs='+',
                        default=custom_defaults['coverage'],
                        help='the expected coverages used to evaluated the accuracies after abstention')

    parser.add_argument('--pretrain_epochs', type=int, default=custom_defaults['pretrain_epochs'])

    parser.add_argument('--method', type=str, choices=['none', 'gamblers'], default=custom_defaults['method'])
    parser.add_argument('--gamblers_temperature', type=float, default=custom_defaults['gamblers_temperature'],
                        help='temperature for the gamblers loss (o in the original paper). '
                            'This should be >1 and <= number of classes')

    parser.add_argument('--use_test', type=bool, default=custom_defaults['use_test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    run_name = f"{args.method}"
    if args.method == 'gamblers':
        run_name += f"_temp{args.gamblers_temperature}"
        
    run_name += f"_lr{args.lr}"
    run_name += f"_wd{args.weight_decay}"
    run_name += f"_tbs{args.train_batch_size}"

    if args.method == 'gamblers':
        run_name += f"_temp{args.pretrain_epochs}"


    wandb.init(
        project="awareness",
        config=vars(args),
        name='cifar100_'+run_name,
        tags=[f"{args.method}"]
    )

    num_classes = 100

    model = resnet101(weights=None, num_classes=num_classes+int(args.method == 'gamblers')).to(device)
    # model = resnet110().to(device);

    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = Compose([
        ToTensor(),
        Normalize( (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = CIFAR100(root='/raid/a.cossu/datasets', train=True, download=True,
                            transform=transform_train)
    test_dataset = CIFAR100(root='/raid/a.cossu/datasets', train=False, download=True,
                        transform=transform_test)

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [0.7, 0.3])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.train_batch_size,
                                            shuffle=True,
                                            drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=args.eval_batch_size,
                                            shuffle=False,
                                            drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.eval_batch_size,
                                            shuffle=False,
                                            drop_last=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr = args.lr

    if args.method == 'gamblers':
        train_criterion = GamblersLoss(args.gamblers_temperature)
    elif args.method == 'none':
        train_criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Wrong method name.")
    pretrain_criterion = torch.nn.CrossEntropyLoss()

    if args.patience > 0:
        stopping = EarlyStopping(mode='max', patience=args.patience)

    if(load_model):
        model = torch.load(model_to_load)
    else:
        for epoch in range(args.epochs):
            train_loss, train_acc = 0., 0.
            model.train()
            for i, (x, y) in enumerate(tqdm(train_loader)):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)

                if args.pretrain_epochs > epoch:
                    loss = pretrain_criterion(out[:, :num_classes], y)
                else:
                    loss = train_criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += (out[:, :num_classes].argmax(dim=-1) == y).sum().item() / float(x.shape[0])

            train_acc /= float(len(train_loader))
            train_loss /= float(len(train_loader))
            wandb.log({"train/acc": train_acc, "train/loss": train_loss}, commit=False)

            if args.pretrain_epochs <= epoch and (epoch+1) % args.metrics_epoch_frequency == 0:
                confidences, predictions, losses = compute_dataset_confidences_predictions(model, args.method, valid_loader, train_criterion, device)
                table = wandb.Table(data=[[s] for s in confidences.cpu().numpy().tolist()], columns=["confidence"])
                wandb.log({f"valid_confidence/confidence_{epoch}": wandb.plot.histogram(table, "confidence", title=f"Valid confidence epoch {epoch}")},
                        commit=False)
                for conf in args.eval_confidence_threshold:
                    valid_acc, valid_loss, num_predictions = evaluate_with_threshold(confidences, predictions, losses, confidence_threshold=conf)
                    wandb.log({f"valid{conf}/acc": valid_acc,
                            f"valid{conf}/loss": valid_loss,
                            f"valid{conf}/num_predictions": num_predictions,
                            f"valid{conf}/perc_predictions": 100 * (num_predictions / float(len(valid_dataset)))}, commit=False)
                    if conf == 0.0 and args.patience > 0:
                        stopping.update(valid_acc)

                coverages = evaluate_coverage(confidences, predictions, args.coverage)
                coverages_table = wandb.Table(data=[[x, y] for x, y in coverages.items()], columns=["coverage", "acc"])
                wandb.log({f"valid_coverage": wandb.plot.line(coverages_table, "coverage", "acc", title=f"Valid Coverage")}, commit=False)

                calibration = evaluate_calibration(confidences, predictions, bins=args.eval_confidence_threshold)
                wandb.log({"valid_calibration": calibration}, commit=False)

            wandb.log({"epoch": epoch})

            if args.patience > 0 and stopping.stopped:
                print("Stopping training after epoch ", epoch+1)
                break

    if args.use_test:
        confidences, predictions, losses = compute_dataset_confidences_predictions(model, args.method, test_loader, train_criterion, device)
        table = wandb.Table(data=[[s] for s in confidences.cpu().numpy().tolist()], columns=["confidence"])
        wandb.log({f"test_confidence/confidence": wandb.plot.histogram(table, "confidence", title=f"Test confidence")}, commit=False)

        coverages = evaluate_coverage(confidences, predictions, args.coverage)
        coverages_table = wandb.Table(data=[[x, y] for x, y in coverages.items()], columns=["coverage", "acc"])
        wandb.log({f"test_coverage/test_coverage": wandb.plot.line(coverages_table, "coverage", "acc",
                                                                title=f"Test Coverage")}, commit=False)

        for conf in args.eval_confidence_threshold:
            test_acc, test_loss, num_predictions = evaluate_with_threshold(confidences, predictions, losses, confidence_threshold=conf)
            wandb.log({f"test_{conf}/acc": test_acc,
                    f"test{conf}/loss": valid_loss,
                    f"test_{conf}/num_predictions": num_predictions,
                    f"test_{conf}/perc_predictions": 100 * (num_predictions / float(len(valid_dataset)))},
                    commit=False)

        calibration = evaluate_calibration(confidences, predictions, bins=args.eval_confidence_threshold)
        wandb.log({"test_calibration": calibration})

    if(save_model):
        torch.save(model.state_dict(), 'model'+run_name+'.pth')

    wandb.finish()
