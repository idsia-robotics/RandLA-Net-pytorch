import pickle
from datetime import datetime

import mlflow
import torch
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

from .dataset import RandlanetDataset
from .model import RandlaNet
from .sampler import RandlanetWeightedSampler
from .utils import MODEL_SAVES_PATH, check_create_folder, separated_multi_auc


def train_model(model, max_epochs, train_loader, test_loader, device,
                output_path, checkpoint_path, lr, use_mlflow, n_layers, n_classes, ith_kfold=None):
    """
        Function used to train a model

    Args:
        model: PyTorch model used being trained
        max_epochs: maximum number of epochs
        train_loader: PyTorch Dataloader for train set
        test_loader: PyTorch Dataloader for test/validation set
        device: PyTorch computing device (e.g. 'cpu','cuda')
        output_path: where to save training information and plots
        checkpoint_path: where to save model checkpoint
        lr: learning rate
        use_mlflow: if True, logs metrics to mlflow

    """
    class_weight = torch.tensor(
        list(train_loader.dataset.total_class_count.values())).to(device)
    class_weight = class_weight.float()/torch.sum(class_weight).float()
    class_weight = 1 / (class_weight+0.02)
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='max',
                                  verbose=True,
                                  patience=1,
                                  cooldown=2,
                                  factor=0.95
                                  )
    max_patience = max(max_epochs, 1)
    epochs_logger = tqdm(range(1, max_epochs + 1), desc="epoch")
    num_labels = len(test_loader.dataset.mapping)
    print("Saving Metadata")
    inv_map = {v: k for k, v in test_loader.dataset.mapping.items()}
    print(inv_map)
    metadata = dict()
    metadata['label_mapping'] = test_loader.dataset.mapping
    metadata['inv_map'] = inv_map
    metadata['best_epoch'] = -1
    index_best = -1
    best_checkpoint_path = ''
    with open(f'{output_path}metadata.pkl', 'wb') as file:
        pickle.dump(metadata, file)
    if use_mlflow:
        mlflow.set_tracking_uri("http://localhost:9999")
        mlflow.set_experiment('randlanet')
        mlflow.start_run()
        ml_flow_run_id = mlflow.active_run().info.run_id
        if ith_kfold is not None:
            mlflow.log_param("k-fold iteration", ith_kfold)
        mlflow.log_param("device", device)
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("max_patience", max_patience)
    mean_iou_list = [0]
    history = pd.DataFrame()
    print("Start Training")

    for epoch in epochs_logger:
        # Training
        train_loss, train_acc = train_epoch(device, loss, model,
                                            optimizer, train_loader, n_layers,
                                            class_weight)
        val_acc, val_iou, val_mean_iou, val_aucs, val_mean_auc = validation(device, model, test_loader, n_layers,
                                                                            n_classes, scheduler)

        if val_mean_iou > np.max(mean_iou_list):
            index_best = len(mean_iou_list)
        mean_iou_list.append(val_mean_iou)
        checkpoint_name = checkpoint_save(checkpoint_path, epoch, val_acc, val_mean_iou,
                                          model)

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        iou_dict = {f"iou_{inv_map[i]}": val_iou[i]
                        for i in range(len(val_iou))}
        auc_dict = {f"auc_{inv_map[i]}": val_aucs[i]
                        for i in range(len(val_aucs))}

        history = history.append({"epoch": epoch,
                                  "train_loss": train_loss,
                                  "train_av_acc": train_acc,
                                  "val_av_acc": val_acc,
                                  "val_av_iou": val_mean_iou,
                                  "val_auc": val_mean_auc,
                                  **iou_dict,
                                  **auc_dict}, ignore_index=True)


        if use_mlflow:
            mlflow.log_metric("train_loss", train_loss, epoch)
            mlflow.log_metric("train_acc", train_acc, epoch)
            mlflow.log_metric("val_acc", val_acc, epoch)
            mlflow.log_metric("val_mean_iou", val_mean_iou, epoch)
            mlflow.log_metric("val_mean_auc", val_mean_auc, epoch)
            
            mlflow.log_metrics(iou_dict, epoch)
            mlflow.log_metrics(auc_dict, epoch)
            mlflow.log_metric("lr", current_lr, epoch)

        if metadata['best_epoch'] != index_best:
            metadata['best_epoch'] = index_best
            best_checkpoint_path = checkpoint_name
            if use_mlflow:
                mlflow.log_param("best_epoch", index_best)
                mlflow.log_param("best_epoch_checkpoint_path", best_checkpoint_path)
            with open(f'{output_path}metadata.pkl', 'wb') as file:
                pickle.dump(metadata, file)
        if epoch - index_best > max_patience:
            print("\n\nearly stopping!")
            break
        epochs_logger.set_postfix_str(f"t_loss={train_loss:.5f}, "
                                      f"t_acc={train_acc:.5f}, "
                                      f"v_acc={val_acc:.5f}, "
                                      f"v_iou={val_mean_iou:.5f}")
    history_save_path = f"{output_path}history.csv"
    history.to_csv(history_save_path)
    print(f"best epoch:{index_best}")
    print("Finished Training")
    if use_mlflow:
        mlflow.end_run()
        return best_checkpoint_path, history_save_path, ml_flow_run_id
    else:
        return best_checkpoint_path, history_save_path, None



def train_epoch(device, loss_function, model, optimizer,
                train_loader, n_layers, class_weight):
    """
        Function that train a single epoch

    Args:
        device: PyTorch computing device (e.g. 'cpu','cuda')
        loss_function: PyTorch loss function
        model: PyTorch model used being trained
        optimizer: PyTorch optimizer used for training
        train_loader: PyTorch Dataloader for train set

    Returns:
        updated history dictionary

    """
    train_logger = tqdm(train_loader,
                        desc="Train",
                        total=len(train_loader))

    train_losses = []
    train_accs = []
    model.train()

    for input_list in train_logger:
        inputs = unpack_input(input_list, n_layers, device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        labels = torch.reshape(inputs['labels'], [-1])

        one_hot = torch.zeros_like(outputs)
        one_hot[range(labels.shape[0]), labels] = 1
        one_hot = one_hot * class_weight
        one_hot = torch.sum(one_hot, dim=1)
        loss = loss_function(outputs, labels)
        loss = loss*one_hot
        loss = loss.mean()
        loss.backward()
        preds = F.log_softmax(outputs, dim=-1).argmax(-1)
        train_acc = (preds == labels).to(torch.float32).mean()
        optimizer.step()

        train_logger.set_postfix_str(f"t_loss={loss.item():.5f}, "
                                     f"t_acc={train_acc.item():.5f}")

        train_losses.append(loss.item())
        train_accs.append(train_acc.item())

    return np.mean(train_losses), np.mean(train_accs)


def validation(device, model, test_loader, n_layers, n_classes, scheduler):
    """
        Given a model a dataset and a set of parameters, the function returns
        all the metrics relative to the validation set

    Args:
        device: PyTorch computing device (e.g. 'cpu','cuda')
        model: PyTorch model to be validated
        test_loader: PyTorch Dataloader for test/validation set

    Returns:
        updated history dictionary and multiple validation metrics

    """
    model.eval()
    gt_classes = [0 for _ in range(n_classes)]
    positive_classes = [0 for _ in range(n_classes)]
    true_positive_classes = [0 for _ in range(n_classes)]
    val_total_correct = 0
    val_total_seen = 0
    all_preds = torch.Tensor().cpu()
    all_gts = torch.Tensor().cpu()
    test_logger = tqdm(test_loader,
                       desc="Validation",
                       total=len(test_loader))
    auc_every = len(test_loader)//5
    with torch.no_grad():
        for cnt, input_list in enumerate(test_logger):
            inputs = unpack_input(input_list, n_layers, device)
            outputs = model(inputs)
            logits = F.log_softmax(outputs, dim=-1)
            pred = logits.argmax(1).cpu().numpy()
            labels = torch.reshape(inputs['labels'], [-1]).cpu()
            if cnt % auc_every == 0:
                all_gts = torch.cat([all_gts, labels])
                all_preds = torch.cat([all_preds, logits.cpu()])
            labels = labels.numpy()
            correct = np.sum(pred == labels)
            val_total_correct += correct
            val_total_seen += len(labels)

            conf_matrix = confusion_matrix(labels, pred,
                                           np.arange(0, n_classes, 1))
            gt_classes += np.sum(conf_matrix, axis=1)
            positive_classes += np.sum(conf_matrix, axis=0)
            true_positive_classes += np.diagonal(conf_matrix)

    iou_list = []
    for n in range(0, n_classes, 1):
        iou = true_positive_classes[n] / float(
            gt_classes[n] + positive_classes[n] - true_positive_classes[n])
        iou_list.append(iou)
    mean_iou = sum(iou_list) / float(n_classes)
    val_acc = val_total_correct / float(val_total_seen)
    val_aucs = separated_multi_auc(
        pred=all_preds, label=all_gts, num_labels=n_classes)
    mean_val_auc = np.mean(list(val_aucs.values()))
    print('eval accuracy: {}'.format(val_acc))
    print('mean IOU:{}'.format(mean_iou))
    print('mean AUC:{}'.format(mean_val_auc))
    # update the lr scheduler step
    # scheduler.step()
    scheduler.step(mean_iou)
    #mean_iou = 100 * mean_iou
    print('Mean IoU = {:.1f}%'.format(100*mean_iou))
    s = '{:5.2f} | '.format(100*mean_iou)
    for IoU in iou_list:
        s += '{:5.2f} '.format(100 * IoU)
    print('-' * len(s))
    print(s)
    print('-' * len(s) + '\n')
    return val_acc, iou_list, mean_iou, val_aucs, mean_val_auc


def unpack_input(input_list, n_layers, device):
    inputs = dict()
    inputs['xyz'] = input_list[:n_layers]
    inputs['neigh_idx'] = input_list[n_layers: 2 * n_layers]
    inputs['sub_idx'] = input_list[2 * n_layers:3 * n_layers]
    inputs['interp_idx'] = input_list[3 * n_layers:4 * n_layers]
    for key, val in inputs.items():
        inputs[key] = [x.to(device) for x in val]
    inputs['features'] = input_list[4 * n_layers].to(device)
    inputs['labels'] = input_list[4 * n_layers + 1].to(device)
    inputs['input_inds'] = input_list[4 * n_layers + 2].to(device)
    inputs['cloud_inds'] = input_list[4 * n_layers + 3].to(device)
    return inputs


def checkpoint_save(checkpoint_path, epoch, mean_v_acc,
                    mean_v_iou, model):
    """
        This function saves a model checkpoint using PyTorch formats

    Args:
        checkpoint_path: where to save model checkpoint
        epoch: epoch number
        mean_v_acc: mean validation accuracy (if only one per epoch than that
            value is used)
        mean_val_loss: mean validation loss (if only one per epoch than that
            value is used)
        mean_v_iou: mean validation iou (if only one per epoch than that
            value is used)
        model: PyTorch model to be saved

    """
    checkpoint_name = f"{epoch}" \
                      f"_v_acc={mean_v_acc:.3f}" \
                      f"_v_iou={mean_v_iou}" \
                      f"_state_dict.pth"
    checkpoint_filename = f"{checkpoint_path}{checkpoint_name}"
    torch.save(model.state_dict(), checkpoint_filename)
    return checkpoint_filename


def train_randlanet_model(train_set_list, test_set_list, hyperpars, use_mlflow=False,
                          num_workers=4, model_name=None):
    """
        Function for training randlanet using provided point clouds filepath
        as train and test sets. Logs metrics to mlflow if use_mlflow==True.

    Args:
        train_set_list: list of path to full pc folders to use as train
        test_set_list: list of path to full pc folders to use as test
        use_mlflow: if True, logs metrics to mlflow
        max_epochs: maximum number of epochs
        batch_size: batch size
        num_workers: number of parallel pytorch workers to load data
        learning_rate: learning rate for training
        model_name: name of the model folder. If None, timestamp is used

    """

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    available_gpu = "cuda:0"
    device = torch.device(available_gpu if use_cuda else "cpu")

    # Parameters for pytorch dataloaders
    train_params = {"batch_size": hyperpars['batch_size'],
                    "shuffle": False,
                    "num_workers": num_workers,
                    "pin_memory": False}
    test_params = {"batch_size": hyperpars['val_batch_size'],
                   "shuffle": False,
                   "num_workers": num_workers,
                   "pin_memory": False}

    if model_name is None:
        model_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    model_save_folder = (MODEL_SAVES_PATH + f'{model_name}/')
    output_path = f"{model_save_folder}output/"
    checkpoint_path = f"{model_save_folder}checkpoints/"
    check_create_folder(output_path)
    check_create_folder(checkpoint_path)

    train_set = RandlanetDataset(train_set_list, **hyperpars)
    train_sampler = RandlanetWeightedSampler(
        train_set, hyperpars['batch_size'] * hyperpars['train_steps'])
    train_loader = data.DataLoader(
        train_set, sampler=train_sampler, **train_params)
    test_set = RandlanetDataset(test_set_list, **hyperpars)
    test_sampler = RandlanetWeightedSampler(
        test_set, hyperpars['val_batch_size'] * hyperpars['val_steps'])
    test_loader = data.DataLoader(
        test_set, sampler=test_sampler, **test_params)

    with open(f"{output_path}datasets_used.txt", "a") as fl:
        fl.write(f'Datasets used:\n'
                 f'     train: {train_set_list}\n'
                 f'     test: {test_set_list}')
    Warning("Re mapping of labels values from original to 0 to max_num_labels")

    model = RandlaNet(
        hyperpars['d_out'], hyperpars['num_layers'], hyperpars['num_classes'])
    model = model.to(device)
    train_model(model, hyperpars['max_epoch'], train_loader, test_loader, device,
                output_path, checkpoint_path, hyperpars['learning_rate'], use_mlflow,
                hyperpars['num_layers'], hyperpars['num_classes'])
