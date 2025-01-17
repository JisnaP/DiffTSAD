import json
import torch
from datetime import datetime
import torch.nn as nn
import os
from plotting import *
from args_ld import get_parser
from utils import *
from LatentDiffusionModel import LatentDiffusion
from predict_anomalies import Predictor
from train import Trainer
from anomaly_scores_loader import AnomalyScoreLoader
if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    print(args_summary)

    if dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        (X_train, _), (X_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}'
        (X_train, _), (X_test, y_test) = get_data(dataset, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    n_features = X_train.shape[1]
    print(f"X_train shape:{X_train.shape}")
    target_dims = get_target_dims(dataset)
    
    out_dim = n_features
    N=X_train.shape[0]
    train_dataset = SlidingWindowDataset(X_train, window_size, horizon=1,stride=1)
    test_dataset = SlidingWindowDataset(X_test, window_size, horizon=1,stride=1)

    train_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, shuffle=False, test_dataset=test_dataset
    )
    anomaly_loader = AnomalyScoreLoader(window_size=window_size, N=N)
    file_path_train = 'DiffTSAD/output/SMD/1-1/anomaly_scores/train/anomaly_scores.pkl'
    anomaly_scores_tensor_train = anomaly_loader.load_anomaly_scores(file_path_train)
    
    file_path_test = 'DiffTSAD/output/SMD/1-1/anomaly_scores/test/anomaly_scores.pkl'
    anomaly_scores_tensor_test = anomaly_loader.load_anomaly_scores(file_path_test)
    
    # Normalize anomaly scores
    normalized_anomaly_scores_train = anomaly_loader.normalize_anomaly_scores(anomaly_scores_tensor_train)
    normalized_anomaly_scores_test = anomaly_loader.normalize_anomaly_scores(anomaly_scores_tensor_test)
    
    # Check the shapes and values after normalization
    print(f"Normalized anomaly scores (train) shape: {normalized_anomaly_scores_train.shape}")
    print(f"Min value (train) after normalization: {normalized_anomaly_scores_train.min()}")
    print(f"Max value (train) after normalization: {normalized_anomaly_scores_train.max()}")
    # Create anomaly_score Sliding window dataset
    anomaly_score_dataset_train,anomaly_score_dataset_test=anomaly_loader.create_anomaly_score_dataset(
      normalized_anomaly_scores_train,normalized_anomaly_scores_test=normalized_anomaly_scores_test)
    
    # Create DataLoaders
    train_anomaly_score_loader, test_anomaly_score_loader = anomaly_loader.create_anomalyscores_loaders(
      anomaly_score_dataset_train, 
      batch_size, shuffle=False, 
      anomaly_score_dataset_test=anomaly_score_dataset_test
      )
    print(f"Train DataLoader created with batch size: {batch_size}")
    if test_loader is not None:
        print(f"Test DataLoader created with batch size: {batch_size}")
    for i,(X,y) in enumerate(train_anomaly_score_loader):
      print(f"Shape of anomaly_score loaded:{X.shape}")
      if i<1:
        break
    
    for i,(X,y) in enumerate(train_loader):
      print(f"shape of X_train loaded:{X.shape}")
      if i<1:
        break



    model = LatentDiffusion(
        n_features=n_features,
        batch_size=batch_size,
        window_size=window_size,
        out_dim=out_dim,
        time_steps=1,
        noise_steps=1,
        denoise_steps=1,
        dim=64,
        init_dim=64,
        dim_mults=(1,2,4),
        channels=24,
        groups=8,
        gru_n_layers=1,
        n_layers=3,
        schedule=args.schedule,
        gru_hid_dim=args.gru_hid_dim,
        kernel_size=args.kernel_size,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        use_gatv2=args.use_gatv2, 
        alpha=args.alpha
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    
    

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    trainer.fit(train_loader,train_anomaly_score_loader )

    plot_losses(trainer.losses, dataset)

    # Check test loss
    test_loss = trainer.evaluate(test_loader,test_anomaly_score_loader)
    
    print(f"Test loss: {test_loss:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )
    stride=1
    N_windows=(N-window_size)/stride +1
    #print(f"N_windows:{N_windows}")
    dropped_points = N_windows % batch_size
    dropped_points=int(dropped_points)
    label = y_test[window_size-1:N-dropped_points] if y_test is not None else None
    #label = y_test[window_size:] 
    predictor.predict_anomalies(X_train, X_test,label,normalized_anomaly_scores_train,normalized_anomaly_scores_test)
    plotter(args.dataset,X_test, predictor.recons, predictor.df.to_numpy() ,label)

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
