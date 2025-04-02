def train_pseudo_history(args, train_loader, val_dataloader, test_dataloader, model, ema_model1, ema_model2, optimizer, save_pseudo_path, l_dl, u_dl):

    recorder1 = [[] for i in range(train_loader.dataset.__len__())]
    recorder2 = [[] for i in range(train_loader.dataset.__len__())]
    alpha = args.alpha
    beta = args.beta  
    best_loss = np.inf
    best = 0
    curr_iter = 0
    global lr
    iters = []

    class_means = torch.zeros(n_class)
    class_stds = torch.ones(n_class)
    class_counts = torch.zeros(n_class)

    model.train()

    bar = tqdm(l_dl)
    for (batch_lb, batch_ulb) in tqdm(zip(l_dl, u_dl)):
        img_lb_w, img_lb_s, label_lb = batch_lb['weak_image'].cuda(), batch_lb['strong_image'].cuda(), batch_lb['label'].cuda()
        index, img_ulb_w, img_ulb_s, label_ulb = batch_ulb['index'], batch_ulb['weak_image'].cuda(), batch_ulb['strong_image'].cuda(), batch_ulb['label'].cuda()
        
        l_image = torch.cat([img_lb_w, img_lb_s])
        l_target = torch.cat([label_lb, label_lb])
        u_image = torch.cat([img_ulb_w, img_ulb_s])
        u_target = label_ulb
        
        # Calculate loss for labeled data
        feat_l, logits = model(l_image)
        ce = CE_mean(logits, l_target.float())
        
        # Calculate loss for unlabeled data
        feat_u, logits_u = model(u_image)
        logits_u_w, logits_u_s = logits_u.chunk(2)
        feat_u_w, feat_u_s = feat_u.chunk(2)
        
        # Calculate pseudo labels and their corresponding loss values
        probs = torch.softmax(logits_u_w, dim=-1)
        max_probs, pseudo_labels = torch.max(probs, dim=-1)

        with torch.no_grad():
            feat_ema, logits_ema = ema_model1(img_ulb_w)
            probs_ema = torch.softmax(logits_ema, dim=-1)
            probs = (probs + probs_ema) / 2.0
            max_probs, pseudo_labels = torch.max(probs, dim=-1)
        
        strong_probs = torch.softmax(logits_u_s, dim=-1)
        max_probs_s, targets_u_s = torch.max(strong_probs, dim=-1)
        
        loss_values = F.cross_entropy(logits_u_s, pseudo_labels, reduction='none')
        
        # Dynamic threshold based on loss and confidence
        model.eval()
        with torch.no_grad():
            for fwd_pass in range(5):
                enable_dropout(model)
                _, output = model(u_image)
                output = torch.softmax(output, dim=-1)
                output = output / 0.95
                output_w, ouput_s = output.chunk(2)
                loss_drop = F.cross_entropy(ouput_s, pseudo_labels, reduction='none')
                
                np_output = ouput_s.detach().cpu().numpy()[np.newaxis, :, :]
                np_loss = loss_drop.detach().cpu().numpy()[np.newaxis, :]
                if fwd_pass == 0:
                    dropout_pred = np_output
                    dropout_loss = np_loss
                else:
                    dropout_pred = np.vstack((dropout_pred, np_output))
                    dropout_loss = np.vstack((dropout_loss, np_loss))   
                  
            epsilon = sys.float_info.min

            mean_pred = np.mean(dropout_pred, axis=0) + epsilon
            mean_loss = np.mean(dropout_loss, axis=0) + epsilon

            entropy = -np.sum(mean_pred * np.log(mean_pred), axis=-1)
            entropy[np.isnan(entropy)] = 0
            entropy = torch.tensor(entropy).cuda()
        
        model.train()
        
        loss_threshold = mean_loss.mean() + mean_loss.std()
        confidence_threshold = mean_pred.mean() + mean_pred.std()

        # Mask for high confidence and low loss samples
        clean_mask = (max_probs_s > confidence_threshold) & (loss_values < loss_threshold) 
        noise_mask = ~clean_mask
 
        if torch.any(noise_mask):
            mixed_image, y_a, y_b, lam= mixup_tru_unc(l_image, img_ulb_s[noise_mask], l_target, probs[noise_mask])
            feat, output_mix = model(mixed_image)
            mix_loss = mixup_criterion(output_mix, y_a, y_b, lam)
        else: 
            mix_loss = torch.tensor(0.0) 
        
        # Weight calculation
        weights = 1.0 / (1.0 + entropy)

        # Select samples
        high_confidence_logits = logits_u_s[clean_mask]
        selected_pseudo_labels = pseudo_labels[clean_mask]
        selected_weights = weights[clean_mask]
        
        # Compute weighted loss
        Lu = (selected_weights  * F.cross_entropy(high_confidence_logits, selected_pseudo_labels, reduction='none')).mean()

        loss = ce + alpha * Lu + beta * mix_loss

        # Update EMA model
        with torch.no_grad():
            activations, logits_ema1 = ema_model1(img_ulb_w)
            activations, logits_ema2 = ema_model2(img_ulb_s)
            pred1 = F.softmax(logits_ema1, dim=1).cpu().data
            pred2 = F.softmax(logits_ema2, dim=1).cpu().data
            
            for i, ind in enumerate(index + label_num):
                current_value1 = np.array(pred1[i].numpy().tolist())
                current_value2 = np.array(pred2[i].numpy().tolist())
                recorder1[ind].append(current_value1.tolist())
                recorder2[ind].append(current_value2.tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model1, 0.9, curr_iter)
        update_ema_variables(model, ema_model2, 0.9, curr_iter)
