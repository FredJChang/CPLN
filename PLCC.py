def history_clean(args, recorder1, recorder2, train_loader, model, label_num, datamaker):
    '''add all avg logits to dataset'''
    
    recorder1_array = np.array([np.array(logits) for logits in recorder1], dtype=object)
    recorder2_array = np.array([np.array(logits) for logits in recorder2], dtype=object)
    logits_combined = 0.5 * np.array(recorder1_array) + 0.5 * np.array(recorder2_array) # only array can add
    
    logits_combined_avg_list = []
    for item in logits_combined:
        if np.isnan(item).all():
            avg_item = np.pad(item, (0, n_class), constant_values=-1) # pad -1 to nan
        else:
            avg_item = np.mean(item, axis=0).tolist() # avg logits
        logits_combined_avg_list.append(avg_item)
    logits_combined_avg_array = np.vstack(logits_combined_avg_list)
    clean_indices = list(set(range(len(datamaker.train_dataset)))-set(range(label_num)))

    # ''' 
    # idea1 SHCS
    # '''
    pseudo_label = {}
    labels_recorder1 = [[np.argmax(prob_dist) for prob_dist in record] for record in recorder1]
    labels_recorder2 = [[np.argmax(prob_dist) for prob_dist in record] for record in recorder2]
    for index in range(len(labels_recorder1)):
        if index > label_num : # except label data
            record_recorder1, record_recorder2 = labels_recorder1[index], labels_recorder2[index]
            if len(record_recorder1) != 0 and len(record_recorder2) != 0: 
                # unlabel data 
                v1 = most_common_value_with_threshold(record_recorder1, args.most_common_threshold)
                v2 = most_common_value_with_threshold(record_recorder2, args.most_common_threshold)
                if v1 is not None and v2 is not None and v1 == v2:
                    pseudo_label[index] = Counter(record_recorder1).most_common(1)[0][0]
                    clean_indices.append(index)
                # if record_recorder1[:] == record_recorder2[:]:
                #     if all(x == record_recorder1[-1] for x in record_recorder1[:]) and all(x == record_recorder2[-1] for x in record_recorder2[:]):
                #         pseudo_label[index] = record_recorder1[-1]
                      
    '''
    idea2 FHCS
    '''
    selected_data_logits1 = {}
    for pseudo_index, pseudo_value in pseudo_label.items():
        selected_data_logits1[pseudo_index] = [row[pseudo_value] for row in recorder1[pseudo_index]]
    average_logits_per_data1 = {key: np.mean(logits, axis=0) for key, logits in selected_data_logits1.items()}
    index1 = np.argsort(list(average_logits_per_data1.values()))
    filter_ratio = args.filter_ratio
    indices1 = index1[int(len(index1) * filter_ratio):]
    clean_indices1 = [list(average_logits_per_data1.keys())[i] for i in indices1]
    
    selected_data_logits2 = {}
    for pseudo_index, pseudo_value in pseudo_label.items():
        selected_data_logits2[pseudo_index] = [row[pseudo_value] for row in recorder2[pseudo_index]]
    average_logits_per_data2 = {key: np.mean(logits, axis=0) for key, logits in selected_data_logits2.items()}
    index = np.argsort(list(average_logits_per_data2.values()))
    filter_ratio = args.filter_ratio
    indices2 = index[int(len(index) * filter_ratio):]
    clean_indices2 = [list(average_logits_per_data2.keys())[i] for i in indices2]
    clean_indices = list(set(clean_indices1) & set(clean_indices2) & set(clean_indices))

    clean_index = clean_indices #+ list(range(label_num)) # add label data
    dataset = datamaker.train_dataset
    real_indices = list(range(len(dataset)))
    
    clean_set = Subset(dataset, clean_index)
    clean_loader = DataLoader(clean_set, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    
    clean_logits = logits_combined_avg_array[clean_index]
    pseudo = np.argmax(clean_logits, axis=1)
    pseudo_one_hot = np.eye(n_class)[pseudo]
    
    clean_loader = DataManager(args, clean_index, pseudo_one_hot).clean_loader # add label data in Datamanager
    
    return clean_loader
