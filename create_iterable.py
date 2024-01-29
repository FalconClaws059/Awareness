import itertools

def cartesian_product_itertools(train_batch_size,val_batch_size, metrics_epoch_frequency, epochs,lr,weight_decay,patience, eval_confidence_threshold, coverage, pretrain_epochs, method, gamblers_temperature, use_test):
    return list(itertools.product(train_batch_size,val_batch_size, metrics_epoch_frequency, epochs,lr,weight_decay,patience, eval_confidence_threshold, coverage, pretrain_epochs, method, gamblers_temperature, use_test))
