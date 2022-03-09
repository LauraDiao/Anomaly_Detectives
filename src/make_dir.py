import os, warnings

warnings.filterwarnings("ignore")


def init_():
    """creates directories needed to run repo"""

    data_path = "data"
    raw_path = "data/raw"
    raw_c_path = "data/raw/train_c"
    raw_r_path = "data/raw/train_r"

    if not os.path.isdir(data_path):
        os.mkdir(data_path)
        
    if not os.path.isdir(raw_path):
        os.mkdir(raw_path)
        os.mkdir(raw_c_path)
        os.mkdir(raw_r_path)

    temp_path = "data/temp"
    temp_c_path = "data/temp/tempdata_c"
    temp_r_path = "data/temp/tempdata_r"

    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)
        os.mkdir(temp_c_path)
        os.mkdir(temp_r_path)

    img_path = "outputs/eda"
    model_img = "outputs/model"

    if not os.path.isdir("model_img"):
        os.mkdir(model_img)

    out_path = "data/out"

    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        os.mkdir(os.path.join(out_path, 'anomaly_detection'))
