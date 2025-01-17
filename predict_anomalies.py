import json
import numpy as np
import torch
from eval_methods import *
from plotting import *
from tqdm import tqdm
from utils import *
import argparse
import datetime
from args import get_parser, str2bool
from utils import *
from LatentDiffusionModel import LatentDiffusion
import os
from anomaly_scores_loader import AnomalyScoreLoader
class Predictor:
    """Latent_Diffusion predictor class.

    :param model: Latent_Diffusion model (pre-trained) used to reconstruct the data
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param pred_args: params for thresholding and predicting anomalies

    """

    def __init__(self, model, window_size, n_features, prediction_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = prediction_args["dataset"]
        self.target_dims = prediction_args["target_dims"]
        self.scale_scores = prediction_args["scale_scores"]
        self.q = prediction_args["q"]
        self.level = prediction_args["level"]
        self.dynamic_pot = prediction_args["dynamic_pot"]
        self.use_mov_av = prediction_args["use_mov_av"]
        self.gamma = prediction_args["gamma"]
        self.reg_level = prediction_args["reg_level"]
        self.save_path = prediction_args["save_path"]
        self.batch_size = 32
        self.use_cuda = True
        self.prediction_args=prediction_args
        self.summary_file_name = summary_file_name
        self.recons=[]
        self.df=None
    def get_score(self, values,normalized_anomaly_scores):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :return np array of anomaly scores + dataframe with reconstruction for each channel and global anomalies
        """

        print("Reconstructing  and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size,horizon=0,stride=1)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,drop_last=True, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        anomaly_score_data=SlidingWindowDataset(normalized_anomaly_scores,self.window_size,horizon=0,stride=1)
        anomaly_score_loader=torch.utils.data.DataLoader(anomaly_score_data,batch_size=self.batch_size,drop_last=True,shuffle=False)
        self.model.eval()
        stride=1
        recons=[]
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)
                (anomaly_scores,_)=next(iter(anomaly_score_loader))
                anomaly_scores=anomaly_scores.permute(0,2,1)
                anomaly_scores=anomaly_scores.to(device)
                _, x_recon = self.model(x,anomaly_scores)
                print(f" x_recon shape:{x_recon.shape}")
                # Extract last reconstruction only
                recons.append(x_recon[:, -1, :].detach().cpu().numpy())
                N=int(values.size(0))
                print(f"N:{N}")
              
                N_windows=(N-self.window_size)/stride +1
                print(f"N_windows:{N_windows}")
                dropped_points = N_windows%self.batch_size
                dropped_points=int(dropped_points)
                print(f"dropped_points:{dropped_points}")
        recons = np.concatenate(recons, axis=0)
        self.recons=recons
        print(f" shape of recons:{self.recons.shape}")
        actual = values.detach().cpu().numpy()[self.window_size-1:(N-dropped_points)]

        
        print(f"shape of actual:{actual.shape}")

        #if self.target_dims is not None:
            #actual = actual[:, self.target_dims]

        anomaly_scores = np.zeros_like(actual)
        df_dict = {}
        for i in range(recons.shape[1]):
            
            df_dict[f"Recon_{i}"] = recons[:, i]
            df_dict[f"True_{i}"] = actual[:, i]
            a_score =  np.sqrt((recons[:, i] - actual[:, i]) ** 2)

            if self.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (1+iqr)

            anomaly_scores[:, i] = a_score
            df_dict[f"A_Score_{i}"] = a_score

        df = pd.DataFrame(df_dict)
        anomaly_scores = np.mean(anomaly_scores, 1)
        df['A_Score_Global'] = anomaly_scores
        self.df=df
        return self.df,self.recons

    def predict_anomalies(self, train, test, true_anomalies,anomaly_scores_train,anomaly_scores_test, load_scores=False, save_output=True,
                          scale_scores=False):
        """ Predicts anomalies

        :param train: 2D array of train multivariate time series data
        :param test: 2D array of test multivariate time series data
        :param true_anomalies: true anomalies of test set, None if not available
        :param save_scores: Whether to save anomaly scores of train and test
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        :param scale_scores: Whether to feature-wise scale anomaly scores
        """

        if load_scores:
            print("Loading anomaly scores")

            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

        else:
            train_pred_df,_ = self.get_score(train,anomaly_scores_train)
            test_pred_df,_ = self.get_score(test,anomaly_scores_test)

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

            train_anomaly_scores = adjust_anomaly_scores(train_anomaly_scores, self.dataset, True, self.window_size)
            test_anomaly_scores = adjust_anomaly_scores(test_anomaly_scores, self.dataset, False, self.window_size)

            # Update df
            train_pred_df['A_Score_Global'] = train_anomaly_scores
            test_pred_df['A_Score_Global'] = test_anomaly_scores

        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

        # Find threshold and predict anomalies at feature-level (for plotting and diagnosis purposes)
        out_dim = self.n_features #if self.target_dims is None else len(self.target_dims)
        all_preds = np.zeros((len(test_pred_df), out_dim))
        for i in range(out_dim):
            train_feature_anom_scores = train_pred_df[f"A_Score_{i}"].values
            test_feature_anom_scores = test_pred_df[f"A_Score_{i}"].values
            epsilon = find_epsilon(train_feature_anom_scores, reg_level=2)

            train_feature_anom_preds = (train_feature_anom_scores >= epsilon).astype(int)
            test_feature_anom_preds = (test_feature_anom_scores >= epsilon).astype(int)

            train_pred_df[f"A_Pred_{i}"] = train_feature_anom_preds
            test_pred_df[f"A_Pred_{i}"] = test_feature_anom_preds

            train_pred_df[f"Thresh_{i}"] = epsilon
            test_pred_df[f"Thresh_{i}"] = epsilon

            all_preds[:, i] = test_feature_anom_preds

        # Global anomalies (entity-level) are predicted using aggregation of anomaly scores across all features
        # These predictions are used to evaluate performance, as true anomalies are labeled at entity-level
        # Evaluate using different threshold methods: brute-force, epsilon and peaks-over-treshold
        e_eval = epsilon_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies, reg_level=self.reg_level)
        p_eval = pot_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies,
                          q=self.q, level=self.level, dynamic=self.dynamic_pot)
        if true_anomalies is not None:
            bf_eval = bf_search(test_anomaly_scores, true_anomalies, start=0.01, end=2, step_num=100, verbose=False)
        else:
            bf_eval = {}

        print(f"Results using epsilon method:\n {e_eval}")
        print(f"Results using peak-over-threshold method:\n {p_eval}")
        print(f"Results using best f1 score search:\n {bf_eval}")

        for k, v in e_eval.items():
            if not type(e_eval[k]) == list:
                e_eval[k] = float(v)
        for k, v in p_eval.items():
            if not type(p_eval[k]) == list:
                p_eval[k] = float(v)
        for k, v in bf_eval.items():
            bf_eval[k] = float(v)

        # Save
        summary = {"epsilon_result": e_eval, "pot_result": p_eval, "bf_result": bf_eval}
        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            json.dump(summary, f, indent=2)

        # Save anomaly predictions made using epsilon method (could be changed to pot or bf-method)
        if save_output:
            global_epsilon = e_eval["threshold"]
            test_pred_df["A_True_Global"] = true_anomalies
            train_pred_df["Thresh_Global"] = global_epsilon
            test_pred_df["Thresh_Global"] = global_epsilon
            train_pred_df[f"A_Pred_Global"] = (train_anomaly_scores >= global_epsilon).astype(int)
            test_preds_global = (test_anomaly_scores >= global_epsilon).astype(int)
            # Adjust predictions according to evaluation strategy
            if true_anomalies is not None:
                test_preds_global = adjust_predicts(None, true_anomalies, global_epsilon, pred=test_preds_global)
            test_pred_df[f"A_Pred_Global"] = test_preds_global

            print(f"Saving output to {self.save_path}/<train/test>_output.pkl")
            train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

        print("-- Done.")
if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument("--model_id", type=str, default=None,
                        help="ID (datetime) of pretrained model to use, '-1' for latest, '-2' for second latest, etc")
    parser.add_argument("--load_scores", type=str2bool, default=False, help="To use already computed anomaly scores")
    parser.add_argument("--save_output", type=str2bool, default=False)
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    if args.model_id is None:
        if dataset == 'SMD':
            dir_path = f"./output/{dataset}/{args.group}"
        else:
            dir_path = f"./output/{dataset}"
        dir_content = os.listdir(dir_path)
        subfolders = [subf for subf in dir_content if os.path.isdir(f"{dir_path}/{subf}") and subf != "logs"]
        date_times = [datetime.datetime.strptime(subf, '%d%m%Y_%H%M%S') for subf in subfolders]
        date_times.sort()
        model_datetime = date_times[-1]
        model_id = model_datetime.strftime('%d%m%Y_%H%M%S')

    else:
        model_id = args.model_id

    if dataset == "SMD":
        model_path = f"./output/{dataset}/{args.group}/{model_id}"
    elif dataset in ['MSL', 'SMAP']:
        model_path = f"./output/{dataset}/{model_id}"
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    # Check that model exist
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # Get configs of model
    print(f'Using model from {model_path}')
    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()
    model_args_path = f"{model_path}/config.txt"

    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)
    window_size = model_args.lookback

    # Check that model is trained on specified dataset
    if args.dataset.lower() != model_args.dataset.lower():
        raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {args.dataset}.")

    elif args.dataset == "SMD" and args.group != model_args.group:
        print(f"Model trained on SMD group {model_args.group}, but asked to predict SMD group {args.group}.")

    window_size = model_args.lookback
    normalize = model_args.normalize
    n_epochs = model_args.epochs
    batch_size = model_args.bs
    init_lr = model_args.init_lr
    shuffle_dataset = model_args.shuffle_dataset
    use_cuda = model_args.use_cuda
    print_every = model_args.print_every
    group_index = model_args.group[0]
    index = model_args.group[2:]
    args_summary = str(model_args.__dict__)

    if dataset == "SMD":
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    else:
        (x_train, _), (x_test, y_test) = get_data(args.dataset, normalize=normalize)

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]
    N=x_train.shape[0]
    target_dims = get_target_dims(args.dataset)
    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, shuffle_dataset, test_dataset=test_dataset
    )

    
    anomaly_loader=AnomalyScoreLoader(window_size,N)
    file_path_train = '/content/drive/MyDrive/DiffTSAD/output/SMD/1-1/anomaly_scores/train/anomaly_scores.pkl'
    anomaly_scores_tensor_train = anomaly_loader.load_anomaly_scores(file_path_train)
    
    file_path_test = '/content/drive/MyDrive/DiffTSAD/output/SMD/1-1/anomaly_scores/test/anomaly_scores.pkl'
    anomaly_scores_tensor_test = anomaly_loader.load_anomaly_scores(file_path_test)
    
    # Normalize anomaly scores
    normalized_anomaly_scores_train = anomaly_loader.normalize_anomaly_scores(anomaly_scores_tensor_train)
    normalized_anomaly_scores_test = anomaly_loader.normalize_anomaly_scores(anomaly_scores_tensor_test)


    model = LatentDiffusion(
        n_features,
        window_size,
        out_dim,
        kernel_size=model_args.kernel_size,
        use_gatv2=model_args.use_gatv2,
        feat_gat_embed_dim=model_args.feat_gat_embed_dim,
        time_gat_embed_dim=model_args.time_gat_embed_dim,
        ld_gru_n_layers=model_args.gru_n_layers,
        ld_gru_hid_dim=model_args.ld_gru_hid_dim, 
        linear_start=model_args.linear_start, 
        linear_end=model_args.linear_end, 
        cosine_s=model_args.cosine_s,
        loss_type=model_args.loss_type,
        conditional=True,
        schedule_opt=None,
        alpha=model_args.alpha,
        dropout=model_args.dropout
    )

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    load(model,f"{model_path}/model.pt", device=device)
    model.to(device)

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
        "save_path": f"{model_path}"
    }
    stride=1
    # Creating a new summary-file each time when new prediction are made with a pre-trained model
    count = 0
    for filename in os.listdir(model_path):
        if filename.startswith("summary"):
            count += 1
    if count == 0:
        summary_file_name = "summary.txt"
    else:
        summary_file_name = f"summary_{count}.txt"
    N_windows=(N-window_size)/stride +1
    #print(f"N_windows:{N_windows}")
    dropped_points = N_windows % batch_size
    dropped_points=int(dropped_points)
    label = y_test[window_size-1:N-dropped_points] if y_test is not None else None
    predictor = Predictor(model, window_size, n_features, prediction_args, summary_file_name=summary_file_name)
    predictor.predict_anomalies(x_train, x_test,normalized_anomaly_scores_train,normalized_anomaly_scores_test,label,
                                load_scores=args.load_scores,
                                save_output=args.save_output)
    plotter(args.dataset, x_test, predictor.recons,predictor.df.to_numpy(), label)
