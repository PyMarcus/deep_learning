import sklearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
tf.random.set_seed(4)


def carrega_transforma(image, label):
    """ Função para carregar e transformar as imagens"""
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels = 3)  # canal = rgb
    return image, label


def prepara_dataset(path, labels, train=True):
    global autotune, resize, batch_size, data_augmentation
    """Função para preparar os dados noo formato do TensorFlow"""
    # Prepara os dados
    image_paths = tf.convert_to_tensor(path)
    labels = tf.convert_to_tensor(labels)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    dataset = dataset.map(lambda image, label: carrega_transforma(image, label))
    dataset = dataset.map(lambda image, label: (resize(image), label), num_parallel_calls=autotune)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    # Se train = True aplica dataset augmentation
    if train:
        dataset = dataset.map(lambda image, label: (data_augmentation(image), label), num_parallel_calls=autotune)
    # Se train = False repete sobre o dataset e retorna
    dataset = dataset.repeat()
    return dataset


def extrai_label(caminho_imagem):
    return caminho_imagem.split("/")[-2]


def ver_nome_da_image():
    print(encoder.inverse_transform(np.argmax(label, axis=1))[0])  # inverte de números para imagem
    plt.imshow((imagem[0].numpy()/255).reshape(224, 224, 3))


def prepara_dataset_validacao():
    dataset_valid = prepara_dataset(X_valid, y_valid, train=False)
    imagem, label = next(iter(dataset_valid))
    return dataset_valid


def cria_modelo():
    modelo = EfficientNetB3(input_shape=(224,224,3), include_top=False)  # remove head do modelo
    #add camadas ao modelo pré (acrescenta no lugar da head)
    modelo = tf.keras.Sequential([
        modelo, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(131, activation="softmax")
    ])
    # sumário do modelo
    modelo.summary()
    # hyperparametros:
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    ep = 1e-7

    # compilacao do modelo:
    modelo.compile(optimizer=Adam(learning_rate=lr,  # adam é a funcao q realiza backpropagation
                                  beta_1=beta1,
                                  beta_2=beta2,
                                  epsilon=ep),
                   loss="categorical_crossentropy",
                   )
    # treinar(modelo)
    """
    modelo.layers[0].trainable = False
    checkpoint = tf.keras.callbacks.ModelCheckpoint("model/melhor_modelo.h5",  # quando encontrar o melhor modelo, salva
                                                    verbose=1,
                                                    save_best=True,
                                                    save_weights_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=4)
    modelo.summary()
    modelo.layers[0].trainable = True
    """
    # Carrega os pesos do ponto de verificação e reavalie
    modelo.load_weights("model/melhor_modelo.h5")
    # dados_teste(modelo)
    faz_previsao("abacaxi.png", modelo, encoder)



def dados_teste(modelo):
    # Carregando e preparando os dados de teste
    camninho_imagens_teste = list(caminho_dados_teste.glob("*/*"))
    imagens_teste = list(map(lambda x: str(x), camninho_imagens_teste))
    imagens_teste_labels = list(map(lambda x: extrai_label(x), imagens_teste))
    imagens_teste_labels = encoder.fit_transform(imagens_teste_labels)
    imagens_teste_labels = tf.keras.utils.to_categorical(imagens_teste_labels)
    test_image_paths = tf.convert_to_tensor(imagens_teste)
    test_image_labels = tf.convert_to_tensor(imagens_teste_labels)
    dataset_teste = (tf.data.Dataset
                     .from_tensor_slices((imagens_teste, imagens_teste_labels))
                     .map(decode_imagens)
                     .batch(batch_size))
    imagem, label = next(iter(dataset_teste))
    print(imagem.shape)
    print(label.shape)
    avaliar_modelo(modelo, dataset_teste)


def decode_imagens(image, label):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, [224,224], method = "bilinear")
    return image, label


def treinar(modelo):
    """
    [PODE LEVAR VÁRIAS HORAS]
    Na cpu, as operações são sequenciais, ou seja, mais lentas
    Se encontrar uma gpu, treina nel"""
    history = modelo.fit(dataset_treino,
                         steps_per_epoch=len(X_treino)//batch_size,
                         epochs=1,
                         validation_data=prepara_dataset_validacao(),
                         validation_steps=len(y_treino)//batch_size)


def avaliar_modelo(modelo, dataset_teste):
    # Avalia o modelo
    loss, acc, prec, rec = modelo.evaluate(dataset_teste)
    print(f"Avaliação:")
    print(f"Precisão: {prec}")
    print(f"Perda: {loss}")


def carrega_nova_imagem(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, [224,224], method = "bilinear")
    plt.imshow(image.numpy()/255)
    image = tf.expand_dims(image, 0)
    return image


def faz_previsao(image_path, model, enc):
    image = carrega_nova_imagem(image_path)
    prediction = model.predict(image)
    pred = np.argmax(prediction, axis = 1)
    return enc.inverse_transform(pred)[0]


def main():
    cria_modelo()


if __name__ == '__main__':
    diretorio_atual = Path.cwd()
    print(diretorio_atual)
    # define caminho para os dados de entrada
    caminho_dados_treino = Path("fruits-360/Training")
    caminho_dados_teste = Path("fruits-360/Test")
    imagens_treino = list(caminho_dados_treino.glob("*/*"))
    # com funções anônimas, é aplicado, a cada item, a função
    imagens_treino = list(map(lambda x: str(x), imagens_treino))
    imagens_treino_labels = list(map(lambda x: extrai_label(x), imagens_treino))
    # converte texto para números para reconhecimento
    encoder = LabelEncoder()
    imagens_treino_labels = encoder.fit_transform(imagens_treino_labels)

    imagens_treino_labels = tf.keras.utils.to_categorical(imagens_treino_labels)
    X_treino, X_valid, y_treino, y_valid = train_test_split(imagens_treino, imagens_treino_labels)
    # Redimensionamento de todas as imagens para 224 x 224
    img_size = 224
    resize = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(img_size, img_size)])
    # Cria o objeto para dataset augmentation
    data_augmentation = tf.keras.Sequential([RandomFlip("horizontal"),
                                             RandomRotation(0.2),
                                             RandomZoom(height_factor=(-0.3, -0.2))])
    # Hiperparâmnetros
    batch_size = 32  # limita a quantidade de imagens, do contrário, estoura memória
    autotune = tf.data.experimental.AUTOTUNE
    # Cria o dataset de treino
    dataset_treino = prepara_dataset(X_treino, y_treino)

    # Shape
    imagem, label = next(iter(dataset_treino))  # 32 imagens, conforme o lot de batch acima
    print(imagem.shape)  # tamanho, pixel larg, pixel alt, canal de cor (RGB)
    print(label.shape)   # 32 labels, 131 = tamanho dos itens na pasta
    ver_nome_da_image()
    prepara_dataset_validacao()
    print(f"FRUTA PASSADA: {main()}")
