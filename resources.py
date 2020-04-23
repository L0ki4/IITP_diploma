import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from dpipe.medim.itertools import extract
import numpy as np

from dpipe.torch.utils import to_np, sequence_to_var
from dpipe.torch.model import optimizer_step


def stratified_loader(ids, labels, models, projections, load_x, load_y):
    labels = np.array(labels)
    ids = np.array(ids)
    train_set = pd.DataFrame(list(zip(ids, labels, models, projections)),
                             columns=['Ids', 'labels', 'models', 'projections'])
    all_diagnoses = list(set(labels))
    all_models = list(set(models))
    all_projections = list(set(projections))
    while True:
        diagn = np.random.choice(all_diagnoses)
        tmgr = np.random.choice(all_models)
        proj = np.random.choice(all_projections)
        # print(diagn, ' ', tmgr, ' ', proj)
        id_ = np.random.choice(list(train_set.Ids[(train_set.labels == diagn) & \
                                                  (train_set.models == tmgr) & (train_set.projections == proj)].values))
        yield (*load_x(id_), load_y(id_))


def stratified_cv_4one(ids, labels, models, tomograph_model, *, val_size=10, n_splits, random_state=17):
    meta = pd.DataFrame(list(zip(ids, labels, models)),
                        columns=['Ids', 'labels', 'models'])
    mask = (meta.models == tomograph_model)
    changed_ids = tuple(np.array(meta.Ids[mask].values))
    changed_labels = tuple(np.array(meta.labels[mask].values))
    train_val_test_ids = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print(f'CV: len(ids) = {len(changed_ids)}')

    for i, (train_val_indices, test_indices) in enumerate(cv.split(changed_ids, changed_labels)):
        train_val_ids = extract(changed_ids, train_val_indices)
        train_val_labels = extract(changed_labels, train_val_indices)
        test_ids = extract(changed_ids, test_indices)
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_size, random_state=25 + i, stratify=train_val_labels)
        train_val_test_ids.append((train_ids, val_ids, test_ids))

    return train_val_test_ids


def stratified_split(ids, labels, models, tomograph_model, *, val_size=10, random_state=17):
    meta = pd.DataFrame(list(zip(ids, labels, models)),
                        columns=['Ids', 'labels', 'models'])
    mask = (meta.models == tomograph_model)
    changed_ids = tuple(np.array(meta.Ids[mask].values))
    changed_labels = tuple(np.array(meta.labels[mask].values))
    train_val_test_ids = []
    print(f'CV: len(ids) = {len(changed_ids)}')

    train_val_ids, test_ids, train_val_labels, _ = train_test_split(changed_ids, changed_labels, test_size=0.3,
                                                                 random_state=656, stratify=changed_labels)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size, random_state=257,
                                          stratify=train_val_labels)
    train_val_test_ids.append((train_ids, val_ids, test_ids))
    print(f'train len = {len(train_ids)}')
    print(f'val len = {len(val_ids)}')
    print(f'test len = {len(test_ids)}')
    return train_val_test_ids


def padded_mri_train_step(*inputs, architecture, criterion, optimizer, n_targets, **optimizer_params) -> np.ndarray:
    architecture.train()
    n_inputs = len(inputs) - n_targets  # in case n_targets == 0

    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:n_inputs], inputs[n_inputs:]

    loss = criterion(architecture(*inputs), *targets)

    optimizer_step(optimizer, loss, **optimizer_params)
    return to_np(loss)


def get_params_string(lr, emb_dim):
    return 'lr =' + str(lr) + ', emb_dim = ' + str(emb_dim)


def get_vlad_params_string(n_clusters, alpha):
    return 'n_clusters =' + str(n_clusters) + ', alpha = ' + str(alpha)
