import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
#from .metrics import ks

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


def ks(y_true, y_prob):
    fpr, tpr, th = roc_curve(y_true, y_prob, pos_label=1)
    return np.max(tpr - fpr)


def model_train(model, trainloader, testloader, MODEL_FILE,
                EPOCHS = 100, ES_PATIENCE = 6, LR = 1e-3,
                FACTOR = 0.1
               ):
    RE_PATIENCE = int(ES_PATIENCE / 2)
    lr = LR
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # 训练过程
    his_train_loss = []
    his_test_loss = []
    best_test_loss = np.inf
    best_test_ks = -np.inf
    es_tol = 0
    re_tol = 0
    loss_eps = ks_eps =  1e-5

    for epoch in list(range(EPOCHS)):
        model.train()
        losses = []
        for it, batch in enumerate(tqdm(trainloader, total = len(trainloader))):
            ids1 = batch['ids1'].to(device)
            ids2 = batch['ids2'].to(device)
            len1 = batch['len1'].to(device)
            len2 = batch['len2'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outs, _, _ = model(ids1, ids2, len1, len2)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = np.mean(losses)
        his_train_loss.append(avg_loss)

        # 在测试集上进行评估
        with torch.no_grad():
            model.eval()
            test_outs_list = []
            test_pos_probs_list = []
            test_losses = []
            test_labels_list = []

            for test_it, test_batch in enumerate(tqdm(testloader, total = len(testloader))):
                test_ids1 = test_batch['ids1'].to(device)
                test_ids2 = test_batch['ids2'].to(device)
                test_len1 = test_batch['len1'].to(device)
                test_len2 = test_batch['len2'].to(device)
                test_labels = test_batch['label'].to(device)
                test_labels_list.append(test_labels.cpu().numpy().squeeze())

                test_outs, _, _ = model(test_ids1, test_ids2, test_len1, test_len2)
                test_loss = criterion(test_outs, test_labels)
                test_pos_probs = torch.softmax(test_outs, dim = 1)[:,1]
                test_pos_probs = test_pos_probs.cpu().numpy().squeeze()
                test_pos_probs_list.append(test_pos_probs)
                test_losses.append(test_loss.item())
                test_outs = test_outs.cpu().numpy().squeeze()
                test_outs_list.append(test_outs)

            avg_test_loss = np.mean(test_losses)
            his_test_loss.append(avg_test_loss)
            test_probs = np.hstack(test_pos_probs_list)
            test_gts = np.hstack(test_labels_list)
            test_ks = ks(test_gts, test_probs)

    #     print('TEST ' * 3, len(test_gts), len(test_probs))
        print("{epoch}/{EPOCHS} - train_loss: {avg_loss:.4f} - test_loss: {avg_test_loss:.4f} - test_ks: {test_ks:.4f}".format(
            epoch=epoch, 
            EPOCHS=EPOCHS, 
            avg_loss=avg_loss,
            avg_test_loss=avg_test_loss,
            test_ks=test_ks))

    #     保存最优的模型
        # if test_ks - best_test_ks > ks_eps:
        #     print("Test ks is improved from {best_test_ks:.4f} to {test_ks:.4f}. Model is saved to {MODEL_FILE}".format(
        #         best_test_ks=best_test_ks, test_ks=test_ks, MODEL_FILE=MODEL_FILE))
        #     best_test_ks = test_ks
        #     torch.save({'epoch': epoch,
        #                 'model_state_dict': model.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'train_loss': avg_loss,
        #                 'test_loss': avg_test_loss,
        #                 'test_ks': test_ks
        #             }, MODEL_FILE)

        if best_test_loss - avg_test_loss >= loss_eps:
            print("Test loss is reduced from {best_test_loss:.4f} to {test_loss:.4f}. Model is saved to {MODEL_FILE}".format(
                best_test_loss=best_test_loss, test_loss=avg_test_loss, MODEL_FILE=MODEL_FILE))
            # best_test_loss = avg_test_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_loss,
                        'test_loss': avg_test_loss,
                        'test_ks': test_ks
                       }, MODEL_FILE)

        # 记录测试集loss没有提升的epoch数
        if best_test_loss - avg_test_loss < loss_eps:
            es_tol += 1
            re_tol += 1
            print("test loss is not improved.")
        else:
            best_test_loss = avg_test_loss
            es_tol = 0
            re_tol = 0

        # 如果连续多次，验证集没有提升，加载最新的模型，并降低学习率
        if re_tol >= RE_PATIENCE:
            checkpoint  = torch.load(MODEL_FILE)
            saved_epoch = checkpoint['epoch']
            print("Learning rate is reduced from {:e} to {:e}. Reload model from epoch:{} - {}".format(lr, lr*FACTOR, saved_epoch, MODEL_FILE))
            lr *= FACTOR
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
            re_tol = 0

        if es_tol >= ES_PATIENCE:
            print("Early stopping after {ES_PATIENCE} steps without improvement on testset".format(ES_PATIENCE=ES_PATIENCE))
            break;

    return his_train_loss, his_test_loss

def model_eval(model, dataloader, device = None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        model.eval()
        data_outs_list = []
        data_pos_probs_list = []
    #     data_losses = []
        data_labels_list = []
        for data_it, data_batch in enumerate(tqdm(dataloader, total = len(dataloader))):
            data_ids1 = data_batch['ids1'].to(device)
            data_ids2 = data_batch['ids2'].to(device)
            data_len1 = data_batch['len1'].to(device)
            data_len2 = data_batch['len2'].to(device)
            data_labels = data_batch['label'].to(device)
            data_labels_list.append(data_labels.cpu().numpy().squeeze())

            data_outs, _, _ = model(data_ids1, data_ids2, data_len1, data_len2)
    #         data_loss = criterion(data_outs, data_labels)
    #         data_losses.append(data_loss.item())
            data_pos_probs = torch.softmax(data_outs, dim = 1)[:,1]
            data_pos_probs = data_pos_probs.cpu().numpy().squeeze()
            data_pos_probs_list.append(data_pos_probs)
            data_outs = data_outs.cpu().numpy().squeeze()
            data_outs_list.append(data_outs)

        data_probs = np.hstack(data_pos_probs_list)
        data_gts = np.hstack(data_labels_list)

    data_ks = ks(data_gts, data_probs)
    data_auc = roc_auc_score(data_gts, data_probs)

    return data_ks, data_auc, data_probs, data_gts
