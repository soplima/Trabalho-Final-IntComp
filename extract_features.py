import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import torch

# Modelos de visão baseados em transformers
from transformers import ViTModel, ViTImageProcessor, AutoModel, CLIPImageProcessor

# Modelos via timm (biblioteca com muitos backbones CNN e ViT)
import timm

# Modelo OpenCLIP
import open_clip

# Transformações de imagem
from torchvision import transforms

# Utilitários do timm para configurar preprocessamento automaticamente
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Define dispositivo (GPU se disponível, senão CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(model_name):
    """
    Carrega diferentes modelos de deep learning e seus respectivos preprocessadores.
    """
    # MODELOS ViT (HUGGINGFACE)
    if model_name == "ViT_huge":
        model = ViTModel.from_pretrained("google/vit-huge-patch14-224-in21k")
        feature_extractor = ViTImageProcessor.from_pretrained(
            "google/vit-huge-patch14-224-in21k"
        )
    # VISION TRANSFORMER (timm)
    elif model_name == "vit_base":
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,
        )
        # Obtém configuração de preprocessamento correta do modelo
        data_config = timm.data.resolve_model_data_config(model)
        # Cria pipeline de transformação compatível com o modelo
        feature_extractor = timm.data.create_transform(**data_config, is_training=False)
    elif model_name == "vit_large":
        model = timm.create_model(
            "vit_large_patch16_224", pretrained=True, num_classes=0
        )
        data_config = timm.data.resolve_model_data_config(model)
        feature_extractor = timm.data.create_transform(**data_config, is_training=False)
    # ViT SMALL (HF)
    elif model_name == "ViT_small":
        model = AutoModel.from_pretrained("WinKawaks/vit-small-patch16-224")
        feature_extractor = ViTImageProcessor.from_pretrained(
            "WinKawaks/vit-small-patch16-224"
        )
    # Modelo ResNet50 (timm)
    elif model_name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(model)
        feature_extractor = timm.data.create_transform(**data_config, is_training=False)
    # VITAMIN (timm pretrained)
    elif model_name == "VITAmin":
        model = timm.create_model("vitamin_large_384", pretrained=True, num_classes=0)

        data_config = timm.data.resolve_model_data_config(model)
        feature_extractor = timm.data.create_transform(**data_config, is_training=False)
    # OPENCLIP
    elif model_name == "openclip_vitg14":
        model, _, feature_extractor = open_clip.create_model_and_transforms(
            "ViT-g-14", pretrained="laion2b_s12b_b42k"
        )
    # CNN / MODELOS DIVERSOS (timm)
    elif model_name == "mambaout":
        model = timm.create_model(
            "mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k",
            pretrained=True,
            num_classes=0,
        )
        data_config = timm.data.resolve_model_data_config(model)
        feature_extractor = timm.data.create_transform(**data_config, is_training=False)
    else:
        raise ValueError(f"Modelo {model_name} não suportado.")
    # Coloca modelo em modo de avaliação
    model.eval()
    # Move modelo para GPU/CPU e retorna junto com transformações
    return model.to(device), feature_extractor


def feature_extraction(image_path, model, feature_extractor, model_name):
    """
    transforma uma imagem em um vetor numérico
    """

    image = Image.open(image_path).convert("RGB")
    # Transformers (ViT HF)
    if model_name in ["ViT_huge", "ViT_large", "ViT_base", "ViT_small"]:
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        # média dos tokens da última camada
        features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    # Openclip
    elif "openclip" in model_name:
        image_tensor = feature_extractor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feats = model.encode_image(image_tensor)

        features = feats.squeeze().cpu().numpy()
    # Open Timm com forward_features
    elif hasattr(model, "forward_features"):
        input_tensor = feature_extractor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feats = model.forward_features(input_tensor)
            feats = model.forward_head(feats, pre_logits=True)

        features = feats.squeeze().cpu().numpy()

    # Timm Padrao
    else:
        input_tensor = feature_extractor(image)

        if not torch.is_tensor(input_tensor):
            raise ValueError("A transformação do timm não retornou um tensor.")
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            feats = model(input_tensor)
        features = feats.squeeze().cpu().numpy()
    return features


def features_to_df(folder_path, model, feature_extractor, model_name):
    """
    chama feature_extraction() repetidamente para contruir o dataset completo,
    retornando um DataFrame com as colunas: image_path, label, feature_0, feature_1,
    ..., feature_n
    """
    data = []
    # percorre subpastas
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            label = subfolder
            # percorre imagens da classe
            for image_file in tqdm(
                os.listdir(subfolder_path), desc=f"Processando {subfolder}"
            ):
                image_path = os.path.join(subfolder_path, image_file)

                # filtra extensões válidas
                if image_file.lower().endswith(("png", "jpg", "jpeg")):
                    try:
                        feats = feature_extraction(
                            image_path, model, feature_extractor, model_name
                        )

                        # salva: caminho + label + vetor de features
                        data.append([image_path, label, *feats])

                    except Exception as e:
                        print(f"Erro ao processar a imagem {image_path}: {e}")

    # define número de features dinamicamente
    if data:
        num_features = len(data[0]) - 2
    else:
        num_features = 0

    columns = ["image_path", "label"] + [f"feature_{i}" for i in range(num_features)]

    return pd.DataFrame(data, columns=columns)


def save_dataframe_to_csv(df, save_path, file_name, model_name):
    """
    Salva DataFrame em arquivo CSV.
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    full_path = os.path.join(save_path, file_name + ".csv")

    df.to_csv(full_path, index=False)

    print(f"Arquivos do modelo {model_name} salvos em: {full_path}")


if __name__ == "__main__":
    # modelos que serão testados
    model_choices = ["resnet50", "vit_large"]

    # define diretório base do script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # pasta com imagens
    fonte_folder = os.path.join(BASE_DIR, "Soybean Seeds")

    # pasta onde CSV será salvo
    salvar_folder = BASE_DIR

    # loop por modelos
    for model_name in model_choices:
        try:
            print(f"Processando com o modelo: {model_name}")

            # carrega modelo + preprocessamento
            model, feat_ext = load_models(model_name)

            # extrai features e monta dataset
            df = features_to_df(fonte_folder, model, feat_ext, model_name)

            file_name = f"result_final_{model_name}"

            # salva CSV final
            save_dataframe_to_csv(df, salvar_folder, file_name, model_name)

        except Exception as e:
            print(f"Erro ao processar o modelo {model_name}: {e}")
