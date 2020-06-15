from model import DFL_VGG16,DFL_RESNET,DFL_EfficientNet,MangoNet
from dataset import MangoDataset
from util import *
from tqdm import tqdm
import faiss
from faiss import normalize_L2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
import scipy
BATCH_SIZE=4
trainset = MangoDataset(root = './', train = "train")
valset_all = MangoDataset(root = './', train = "val")
testset = MangoDataset(root = './', train = "test")
# create train/val loaders
train_loader = DataLoader(dataset=trainset,
                          batch_size=BATCH_SIZE, 
                          shuffle=True,
                          num_workers=1)
train_n = len(trainset)
val_n = len(valset_all)

val_loader_all = DataLoader(dataset=valset_all,
                        batch_size=BATCH_SIZE, 
                        shuffle=False,
                        num_workers=1)

valset_new = MangoDataset(root = './', train = "val")
val_loader_new = DataLoader(dataset=valset_new,
                        batch_size=BATCH_SIZE, 
                        shuffle=False,
                        num_workers=1)


test_loader = DataLoader(dataset=testset,
                        batch_size=BATCH_SIZE, 
                        shuffle=False,
                        num_workers=1)

DEVICE = "cuda:1"
net = MangoNet().to(DEVICE)
net.load_state_dict(torch.load("model"))
n_classes = 3

def update_plabels(src_dataloader, tar_dataloader,net,step, k = 50, max_iter = 100):
    embeddings_src,embeddings_tar,labels_src,labels_tar,order=[],[],[],[],[]
    for src_step, (src_input, src_label,_,_,_) in enumerate(src_dataloader):
        src_input = src_input.to(DEVICE)
        src_label = src_label.to(DEVICE)
        feature = torch.mean(net.backbone.extract_features(src_input),axis=(2,3))
        embeddings_src.append(feature.data.cpu())
        labels_src.append(src_label.data.cpu())
    for tar_step, (tar_input, tar_label,_,index,_) in enumerate(tar_dataloader):
#         print(tar_input.shape)
        tar_input = tar_input.to(DEVICE)
        tar_label = tar_label.to(DEVICE)
        feature = torch.mean(net.backbone.extract_features(tar_input),axis=(2,3))
        embeddings_tar.append(feature.data.cpu())
        labels_tar.append(tar_label.data.cpu())
        order.append(index.data.cpu())
    order = np.asarray(torch.cat(order).numpy())
    ind=[]
    for a in range(order.shape[0]):
        ind.append(np.argwhere(a==order)[0][0])
    order = np.int64(ind)
    src_X = np.asarray(torch.cat(embeddings_src).numpy())
    embeddings_src_len = src_X.shape[0]
    tar_X = np.asarray(torch.cat(embeddings_tar).numpy())
    embeddings_tar_len = tar_X.shape[0]
    X = np.concatenate((src_X,tar_X),axis=0)
    print('Updating pseudo-labels...')
    alpha = 0.5
    labels = np.asarray(torch.cat(labels_src).numpy())
    labels_tar = np.asarray(torch.cat(labels_tar).numpy())
    labels = np.concatenate((labels,labels_tar),axis=0)
    labeled_idx = np.asarray([a for a in range(embeddings_src_len)])
    unlabeled_idx = np.asarray([embeddings_src_len+a for a in range(embeddings_tar_len)])
    
    torch.cuda.empty_cache()


    # kNN search for the graph
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

    normalize_L2(X)
    index.add(X) 
    N = X.shape[0]
    Nidx = index.ntotal

    c = time.time()
    D, I = index.search(X, k + 1)
    elapsed = time.time() - c
    print('kNN Search done in %d seconds' % elapsed)

    # Create the graph
    D = D[:,1:] ** 3
    I = I[:,1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - sparse.diags(W.diagonal())
    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N,n_classes))
    A = sparse.eye(Wn.shape[0]) - alpha * Wn
    
    for i in range(n_classes):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] ==i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 / cur_idx.shape[0]
        f, _ = linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:,i] = f

    # Handle numberical errors
    Z[Z < 0] = 0 

    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
    probs_l1[probs_l1 <0] = 0
    entropy = stats.entropy(probs_l1.T)
    weights = 1 - entropy / np.log(n_classes)
    weights = weights / np.max(weights)
#     weights = np.max(probs_l1,1)
    p_labels = np.argmax(probs_l1,1)

    # Compute the accuracy of pseudolabels for statistical purposes
    conf = weights[unlabeled_idx]#>0.2
#     conf[conf<0.5]=0
    conf = (conf-np.min(conf)) /(np.max(conf)-np.min(conf))
    conf = np.log(conf+0.0001)
    conf = (conf-np.min(conf)) /(np.max(conf)-np.min(conf))
#     conf = np.clip(conf,0.3,1)
    pss = p_labels[unlabeled_idx]
    conf_unlabeled_idx = unlabeled_idx[np.argwhere(weights[unlabeled_idx]>0.2).reshape(-1)]
    correct_idx = np.int64(p_labels[unlabeled_idx] == labels[unlabeled_idx])
    
#     print(correct_idx)
    acc = correct_idx.mean()
    print(acc)
    incorrect_idx = np.where(correct_idx==0)
    correct_idx = np.where(correct_idx==1)

    p_labels[labeled_idx] = labels[labeled_idx]
    weights[labeled_idx] = 1.0
#     dataset_tar.train_set.conf=conf[indexs]
#     dataset_tar.train_set.ps=pss[indexs]
    p_weights = weights.tolist()
#     draw_pseudo_label(src_X,tar_X,correct_idx,incorrect_idx,conf,step,"lp")

#     # Compute the weight for each class
#     for i in range(len(self.classes)):
#         cur_idx = np.where(np.asarray(p_labels) == i)[0]
#         self.class_weights[i] = (float(labels.shape[0]) / len(self.classes)) / cur_idx.size
    return conf[order],pss[order],labels_tar[order]
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
def init_center(src_dataloader,classnum):
    net.eval()
    center = torch.zeros((classnum,2560)).to(DEVICE)
    size = torch.zeros((classnum)).to(DEVICE)
    print("init_center")
    for step,(src_input, src_label,_,_,_) in tqdm(enumerate(src_dataloader)):
        src_input = src_input.to(DEVICE)
        src_label = src_label.to(DEVICE).reshape(-1)
        feature = torch.mean(net.backbone.extract_features(src_input),axis=(2,3))
        for a in range(classnum):
            c = (src_label == a).nonzero().reshape(-1)
            size[a] += feature[c].size(0)
            if feature[c].size(0) != 0:
                center[a] += torch.sum(feature[c], axis = 0).detach()
    return center / size.reshape(-1, 1)
def update_plabels_centroid(src_dataloader,tar_dataloader,step):
    c = init_center(src_dataloader,n_classes)
    net.eval()
    print("update_plabels_centroid")
    embeddings_src,embeddings_tar,pss,labels_tar,order,coss=[],[],[],[],[],[]
    for tar_step, (tar_input, tar_label,_,index,_) in tqdm(enumerate(src_dataloader)):
        tar_input = tar_input.to(DEVICE)
        tar_label = tar_label.to(DEVICE)
        feature = torch.mean(net.backbone.extract_features(tar_input),axis=(2,3))
        embeddings_src.append(feature.data.cpu())
    for tar_step, (tar_input, tar_label,_,index,_) in tqdm(enumerate(tar_dataloader)):
        tar_input = tar_input.to(DEVICE)
        tar_label = tar_label.to(DEVICE)
        feature = torch.mean(net.backbone.extract_features(tar_input),axis=(2,3))
        embeddings_tar.append(feature.data.cpu())
        cosine, cos_psuedo_label = torch.max(sim_matrix(feature,c),axis=1)
        coss.append(cosine.data.cpu())
        order.append(index.data.cpu())
        labels_tar.append(tar_label.data.cpu())
        pss.append(cos_psuedo_label.data.cpu())
    labels_tar = np.asarray(torch.cat(labels_tar).numpy())
    pss = np.asarray(torch.cat(pss).numpy())
#     pss = np.argmax(pss,1)
    order = np.asarray(torch.cat(order).numpy())
    ind=[]
    for a in range(order.shape[0]):
        ind.append(np.argwhere(a==order)[0][0])
    order = np.int64(ind)
    coss = np.asarray(torch.cat(coss).numpy())
    coss = (coss-coss.min()) /(coss.max()-coss.min())
    
    
    correct_idx = np.int64(pss == labels_tar)
    incorrect_idx = np.where(correct_idx==0)
    correct_idx = np.where(correct_idx==1)
    
    src_X = np.asarray(torch.cat(embeddings_src).numpy())
    tar_X = np.asarray(torch.cat(embeddings_tar).numpy())

#     draw_pseudo_label(src_X,tar_X,correct_idx,incorrect_idx,coss,step,"cos")
    
    return coss[order],pss[order],labels_tar[order]
subdomain = 5

ce = torch.nn.NLLLoss()
ce_noreduce = torch.nn.NLLLoss(reduce=False)
opt = torch.optim.Adam([{'params': net.parameters(), 'lr': 0.0001}])
EPOCH=2
for sub in range(1,subdomain+1):
#     conf, pss, labeltar = update_plabels_centroid(train_loader, val_loader_all,sub)
    conf, pss, labeltar =update_plabels(train_loader, val_loader_all,net,sub, k = 50, max_iter = 20)
    easy_index = np.argsort(-conf)[:int(conf.shape[0]*(sub/subdomain))].reshape(-1)
    valset_new.conf = conf[easy_index]
    valset_new.ps = pss[easy_index]
    valset_new.imgs = valset_all.imgs[easy_index]
    valset_new.lbls = valset_all.lbls[easy_index]
    val_loader_new = DataLoader(dataset=valset_new,
                            batch_size=BATCH_SIZE, 
                            shuffle=False,
                            num_workers=1)
    
    easycorrect_idx = np.int64(labeltar[easy_index] == pss[easy_index])
    correct_idx = np.int64(labeltar == pss)
    print('easy',easycorrect_idx.mean(),'all',correct_idx.mean())
    
    
    # opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.000005)
    for _ in range(EPOCH):
        net.train()
        iteration=int(train_n/BATCH_SIZE)
        for _ in range(1, iteration+1):
            try:
                src_data, src_label,_,_,_ = src_iter.next()
            except Exception as err:
                src_iter=iter(train_loader)
                src_data, src_label,_,_,_  = src_iter.next()

            try:
                tgt_data, tar_label,conf,_,ps = tgt_iter.next()
            except Exception as err:
                tgt_iter=iter(val_loader_new)
                tgt_data, tar_label,conf,_,ps = tgt_iter.next()


            src_data, src_label = src_data.to(DEVICE), src_label.to(DEVICE)
            tgt_data,tar_label = tgt_data.to(DEVICE),tar_label.to(DEVICE)
            ps_label = ps.to(DEVICE)
            conf_indicator = conf.to(DEVICE)
            
            out1 = net(src_data)
            out1 = F.log_softmax(out1,1)
            loss = ce(out1,src_label)
            
            out1 = net(tgt_data)
            tar_out_NLL = F.log_softmax(out1,1)
            
            target_distance_loss = torch.sum(conf_indicator*ce_noreduce(tar_out_NLL,ps_label))/(torch.sum(conf_indicator)+0.00001)
            
            closs = 0.1*loss + target_distance_loss
            closs.backward()
            opt.step()
            opt.zero_grad()
        
        print(closs)
        net.eval()
        correct = 0
        for data, label,_,_,_ in tqdm(val_loader_all):
            data, label = data.to(DEVICE), label.to(DEVICE)
            out1 = net(data)

            pred = torch.max(out1,axis=1)[1]
            correct += torch.sum((label==pred).int()).detach().item()
        print("valacc:",correct/val_n)
        net.eval()
        correct = 0
        for data, label,_,_,_ in tqdm(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            out1 = net(data)
            pred = torch.max(out1,axis=1)[1]
            correct += torch.sum((label==pred).int()).detach().item()
        print("trainacc:",correct/train_n)
        torch.cuda.empty_cache()