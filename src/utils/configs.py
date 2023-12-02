from datetime import datetime
from ml_collections import ConfigDict

#####################################################################
#                        HYPERPARAMETERS                            #
#####################################################################


def get_path_configs():
    path_cfg = ConfigDict()
    path_cfg["data_dir"] = "data\\HBW\\"
    path_cfg["images_dir"] = "data\\HBW\\images\\val\\"
    path_cfg["yaml_file"] = "data\\HBW\\genders.yaml"
    path_cfg["csv_file"] = "data\\HBW\\dataset.csv"
    path_cfg["smplx_gts"] = "data\\HBW\\smplx\\val\\"
    path_cfg["smplx_model_dir"] = "data\\models\\"
    path_cfg["log_dir"] = "wandb_logs\\"
    path_cfg["checkpoint_dir"] = "model_ckpts\\"
    current_datetime = datetime.now()
    datetime_string = f"{current_datetime.date()}_{current_datetime.hour}-{current_datetime.minute}-{current_datetime.second}"
    path_cfg["checkpoint_file"] = "img2smplx_" + datetime_string
    return path_cfg


def get_data_configs():
    data_cfg = ConfigDict()
    data_cfg["img_size"] = 768
    data_cfg["mean"] = (0.485, 0.456, 0.406)
    data_cfg["std"] = (0.229, 0.224, 0.225)
    data_cfg["strip_thickness"] = 32
    data_cfg["num_heads"] = 1
    data_cfg["num_betas"] = 10
    data_cfg["num_layers"] = 2
    data_cfg["embed_size"] = 256
    data_cfg["hidden_size"] = 512
    data_cfg["img_segment_type"] = "strips"
    return data_cfg


def get_train_configs():
    train_cfg = ConfigDict()
    train_cfg["experiment_name"] = "Run_lr5e-5_dropout0.1"
    train_cfg["batch_size"] = 8
    train_cfg["project_name"] = "HBW Human Shape Estimation"
    train_cfg["epochs"] = 50
    train_cfg["lr"] = 5e-5
    train_cfg["dropout"] = 0.1
    train_cfg["lr_milestones"] = []
    return train_cfg
