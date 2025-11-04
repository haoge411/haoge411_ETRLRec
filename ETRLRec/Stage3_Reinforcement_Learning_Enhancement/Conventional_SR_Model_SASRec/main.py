import os
import time
import torch
import argparse

from .model import SASRec
from .utils import *
from lion_pytorch import Lion
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def Trained_CRM(args, label, candi_sas ,llist, state_dict_path):
    
    dataset = args.user_data_path
    batch_size = args.CRM_batch_size
    
    dataset = data_partition(dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    model = SASRec(usernum, itemnum,args=args).to(args.CRM_device)
    model.eval()
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    if state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(args.CRM_device)))
            tail = state_dict_path[state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            sys.exit()
    model.eval()
    with torch.no_grad():
        sharpen_prob_reward,max_value, max_index,min_value,min_index = evaluate_seq(model,llist,candidate_list=candi_sas,label=label)
    return sharpen_prob_reward,max_value, max_index,min_value,min_index


    


def train_CRM(args):
    u2i_index, i2u_index = build_index(args.user_data_path)
    
    dataset = data_partition(args.user_data_path)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = (len(user_train) - 1) // args.CRM_batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.user_data_path + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.CRM_batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(user_num=usernum, item_num=itemnum, args=args).to(args.CRM_device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    
    
    model.train()
    
    epoch_start_idx = 1
    if args.trained_CRM_path is not None:
        try:
            model.load_state_dict(torch.load(args.trained_CRM_path, map_location=torch.device(args.device)))
            tail = args.trained_CRM_path[args.trained_CRM_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.trained_CRM_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        if epoch % 30 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                folder = args.dataset + '_' + args.train_dir
                fname = 'SA.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))
                print(f"Save the best model to: {os.path.join(folder, fname)}")

            f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SA.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
            print(f"Save the best model to: {os.path.join(folder, fname)}")
    
    f.close()
    sampler.close()
    print("Done")
    # Verify whether the final model has been saved successfully
    if os.path.exists(os.path.join(folder, fname)):
        file_size = os.path.getsize(os.path.join(folder, fname)) / 1024  # KB
        print(f"The final model has been saved successfully! File size: {file_size:.2f} KB")
    else:
        print(f"Error: The final model failed to be saved! The file does not exist: {os.path.join(folder, fname)}")
