#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#instalar api de kaggle para cceder a sus bases de datos sin tener que vajar todos los data sets y cargarlos 
#manualmente
get_ipython().system('pip install -q kaggle')


# In[ ]:


#comando para crear folder
get_ipython().system('mkdir ~/.kaggle')


# In[ ]:


#crear un json en el folder de arriba
get_ipython().system('touch ~/.kaggle/kaggle.json')


# In[ ]:


#token para acceder a la API de kaggle
#{"username":"matassandoval","key":"11dcb6118bc590c8a0bdc00c5d7f54fe"}
api_token = {"username":"matassandoval","key":"11dcb6118bc590c8a0bdc00c5d7f54fe"}


# In[ ]:


import json
#vas a añadir (por eso el "w" de write) el token al json que acabas de crear 
with open("/root/.kaggle/kaggle.json", "w") as file:
    json.dump(api_token, file)


# In[ ]:


#linea de comando que sirve para especificar quienes pueden escribir y modificar el json
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


#bajar el dataset que quieres -> puedes bajar el dataset que quieras usando el link de kaggle como se muestra en 
#el video
get_ipython().system('kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --force')


# In[ ]:


#ESTE COMO QUE NO SIRVIO
#lo de arriba te descargo un zip, ahora extraeras todo en el folder "data"
import zipfile
with zipfile.ZipFile('./chest-xray-pneumonia.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')


# In[ ]:


import glob
import random
import matplotlib.pyplot as plt


# In[ ]:


#funcion para elegir una imagen random 
def get_random_image(dir,condition):
    placeholder=''
    if condition == 'n':
        placeholder='NORMAL'
    elif condition == 'p':
        placeholder='PNEUMONIA'
    else:
        raise Exception("Sorry, invalid condition")
    folder=f'./data/chest_xray/{dir}/{placeholder}/*.jpeg'
    img_paths=glob.glob(folder)
    max_length=len(img_paths)
    randomNumber=random.randint(0,max_length)
    for index, item in enumerate(img_paths, start=1):
        if index == randomNumber:
            print(index,item)
            image = plt.imread(item)
            readyImage=plt.imshow(image)
            return readyImage


# In[ ]:


get_random_image("train","p")


# #image classification algorithm.
# El modelo que se usará es image classification algorithm. Puedes ver su documentación en el link de la imagen.
# 
# El modelo  solo acepta imágenes 224x224, según la documentación de aws. 
# 
# El modelo que se ocupará es de transfer learning. Lo que quiere decir que el modelo ya fue entrenado con otros datos y tiene pesos precargaos, así no tenemos que entrenarlo desde el inicio.
# 
# ____________________________________
# 
# El nombre los archivos de las imagenes de las perosnas con pneumonia empiezan con "person", mientras que las 
# personas sanas, el nombre de los archivos son "IM..."
# 
# Las imagenes de un celular o de una mac generalmente estan en formato RGBA, por lo que tendrías que tendrías que transformarlas a RGB, o sea tendrías que hacer loop por todo tu dataset y convertirlas a RGB. Esto puede ser útil para Gaus.

# In[ ]:


import glob
import matplotlib.pyplot as plt
from PIL import Image

#ira al folder de train y recorrera todos los folders(pneumonia y normal) y todas las imagenes dentro de ese folder
folder=f'./data/chest_xray/train/*/*.jpeg'

#inicio contadores para ir etiquetando las imagenes como "train_pneumonia1", "train_pneumonia2" y así consecutivamente
counterPneu=0
counterNormal=0

#se guardan todos los paths de todas las imagenes de train
img_paths=glob.glob(folder)

for i in img_paths:
    #si se encuentra ese string en el string del path 
    if "person" in i:
        full_size_image=Image.open(i)
        im=full_size_image.resize((224,224))
        #para gaus aqui te haria falta checar como guardarlo de una vez en RGB -> checa documentación de PIL -> Image
        plt.imsave(
            #hasta aqui solo estas diciendo donde se guardara la imagen y con que nombre
            fname='./data/chest_xray/train' + '/train_pneumonia' + str(counterPneu)+'.jpeg'
                    #estas creando un array con la info de la imagen
                   , arr= 
                        #imagen
                        im ,
                        format='jpeg',
                        #si quieres entrenar tu modelo a color, quita esta linea de codigo -> gaus
                        cmap='gray'
        )
        
        counterPneu+=1
    else:
        full_size_image=Image.open(i)
        im=full_size_image.resize((224,224))
        plt.imsave(fname='./data/chest_xray/train' + '/train_normal' + str(counterNormal)+'.jpeg',arr=im,format='jpeg',cmap='gray')
        counterNormal+=1


# ahora hago lo mismo para test y val

# In[ ]:


import glob
import matplotlib.pyplot as plt
from PIL import Image

folder=f'./data/chest_xray/test/*/*.jpeg'

counterPneu=0
counterNormal=0

img_paths=glob.glob(folder)

for i in img_paths:
    if "person" in i:
        full_size_image=Image.open(i)
        im=full_size_image.resize((224,224))
        plt.imsave(fname='./data/chest_xray/test' + '/test_pneumonia' + str(counterPneu)+'.jpeg',arr=im,format='jpeg',cmap='gray')
        counterPneu+=1
    else:
        full_size_image=Image.open(i)
        im=full_size_image.resize((224,224))
        plt.imsave(fname='./data/chest_xray/test' + '/test_normal' + str(counterNormal)+'.jpeg',arr=im,format='jpeg',cmap='gray')
        counterNormal+=1


# In[ ]:


import glob
import matplotlib.pyplot as plt
from PIL import Image

folder=f'./data/chest_xray/val/*/*.jpeg'

counterPneu=0
counterNormal=0

img_paths=glob.glob(folder)

for i in img_paths:
    if "person" in i:
        full_size_image=Image.open(i)
        im=full_size_image.resize((224,224))
        plt.imsave(fname='./data/chest_xray/val' + '/val_pneumonia' + str(counterPneu)+'.jpeg',arr=im,format='jpeg',cmap='gray')
        counterPneu+=1
    else:
        full_size_image=Image.open(i)
        im=full_size_image.resize((224,224))
        plt.imsave(fname='./data/chest_xray/val' + '/val_normal' + str(counterNormal)+'.jpeg',arr=im,format='jpeg',cmap='gray')
        counterNormal+=1


# Crear un dataframe para visualizacion 

# In[ ]:


import glob
import pandas as pd

#vas a acceder a todas las imagenes de todos los folders
folder=f'./data/chest_xray/*/*.jpeg'

#vas a crear una lista donde guardaras todas las categorias, otra para el nombre de archivo y asi
#al final juntaras las 3 listas en un dataframe, con una columna por lista
category=[]
filenames=[]
condition_of_lung=[]

all_files=glob.glob(folder)

#iteras sobre todos los archivos y les vas haciendo append a sus correspondientes listas en orden 
for filename in all_files:
    if "train" in filename:
        if "pneumonia" in filename:
            category.append("train")
            filenames.append(filename)
            condition_of_lung.append("pneumonia")
        elif "normal" in filename:
            category.append("train")
            filenames.append(filename)
            condition_of_lung.append("normal")
    elif "test" in filename:
        if "pneumonia" in filename:
            category.append("test")
            filenames.append(filename)
            condition_of_lung.append("pneumonia")
        elif "normal" in filename:
            category.append("test")
            filenames.append(filename)
            condition_of_lung.append("normal")
    elif "val" in filename:
        if "pneumonia" in filename:
            category.append("val")
            filenames.append(filename)
            condition_of_lung.append("pneumonia")
        elif "normal" in filename:
            category.append("val")
            filenames.append(filename)
            condition_of_lung.append("normal")
#juntas las 3 listas
all_data_df=pd.DataFrame({"dataset type":category,"x-ray result":condition_of_lung,"filename":filenames})

print(all_data_df.head())
            


# In[ ]:


import seaborn as sns

g=sns.catplot(x="x-ray result",col="dataset type",kind="count",palette="ch:.55",data=all_data_df,legend=True)

#es para añadir cuantas imagnes hay exactamente de cada categoria
#recorres las 3 graficas horizontalmente
for i in range(0,3):
    ax=g.facet_axis(0,i)
    #no se como pero aqui pones la cantidad por categoria
    for p in ax.patches:
        ax.text(p.get_x()+0.3,
        p.get_height()*1.05,
        '{0:.0f}'.format(p.get_height()),
        color='black',
        rotation='horizontal',
        size='large')
        
plt.show()


# In[ ]:


import glob
import pandas as pd
import os

#vas a repetir este proceso para los 2 folders, val no porque no entrenara con val
train_folder='./data/chest_xray/train/*.jpeg'
#creas dataframe(ahorita vacio)
train_df_lst=pd.DataFrame(columns=['labels','s3_path'],dtype=object)
#obtienes rodas las imagnes de train
train_imgs_path=glob.glob(train_folder)
#contador para ir iterando sobre el dataframe
counter=0
#argumento para columna de clase
class_arg=''

#recorres todas imagenes y vas agregando al dataframe fila por fila, su clase y path
for i in train_imgs_path:
    if "pneumonia" in i:
        class_arg=1
    else:
        class_arg=0
        #os path.basename es para que solo tome el ultimo argumento del path (img1.jpeg) y no todo el path
        #agregas a cada fila su clase y path base
    train_df_lst.loc[counter]=[class_arg,os.path.basename(i)]
    counter+=1
print(train_df_lst.head())


# In[ ]:


test_folder='./data/chest_xray/test/*.jpeg'
test_df_lst=pd.DataFrame(columns=['labels','s3_path'],dtype=object)
test_imgs_path=glob.glob(test_folder)
counter=0
class_arg=''

for i in test_imgs_path:
    if "pneumonia" in i:
        class_arg=1
    else:
        class_arg=0
    test_df_lst.loc[counter]=[class_arg,os.path.basename(i)]
    counter+=1
print(test_df_lst.head())


# In[ ]:


def save_to_lst(df,prefix):
    #conviertas df a csv
    return df[["labels","s3_path"]].to_csv(
        #sin embargo aqui le das una terminacion .lst en lugar de csv y le das el nombre del train o test
    f"{prefix}.lst",
        #aqui estas separando por tabs
        sep='\t',
        #mantienes los index
        index=True,
        #los lst no tienen header
        header=False
    )

save_to_lst(train_df_lst.copy(),"train")
save_to_lst(test_df_lst.copy(),"test")


# In[ ]:


#una vez que creas tu bucket en s3 guardas estos datos en variables

bucket='x-ray-pneumonia-ml'
print("bucket:{}".format(bucket))
region='us-east-1'
print("region:{}".format(region))
roleArn='arn:aws:s3:::x-ray-pneumonia-ml'
print("roleArn:{}".format(roleArn))


# In[ ]:


#configuras el ambiente donde trabajaras, o sea creo que el bucket donde trabajaras
import os

os.environ["DEFAULT_S3_BUCKET"]=bucket


# Comandos para subir tus imagenes de los folders train y test a tu bucket de S3
# Los argumentos creo que se reciben primero que folder quieres sincroinzar y despues del s3: en que bucket de s3 quieres poner (por eso configuraste arriba el ambiente)

# In[ ]:


get_ipython().system('aws s3 sync ./data/chest_xray/train s3://${DEFAULT_S3_BUCKET}/train/')


# In[ ]:


#boto3, que es una librería para usar s3 con python, mandas tus lst a tu bucket de s3
import boto3

boto3.Session().resource('s3').Bucket(bucket).Object("train.lst").upload_file('./train.lst')
boto3.Session().resource('s3').Bucket(bucket).Object("test.lst").upload_file('./test.lst')


# 6. Llamando al algoritmo
# AWS ECR: Docker que te permite administrar imágenes de tu container
# 
# Estimator object: objeto que llevara las especificaciones (parámetros) para entrenar al modelo
# 
# 

# In[6]:


#boto3, que es una librería para usar s3 con python, mandas tus lst a tu bucket de s3
import boto3


# In[7]:


bucket='x-ray-pneumonia-ml'
print("bucket:{}".format(bucket))
region='us-east-1'
print("region:{}".format(region))
roleArn='arn:aws:s3:::x-ray-pneumonia-ml'
print("roleArn:{}".format(roleArn))


# In[8]:


import sagemaker
from sagemaker import image_uris
import boto3
from sagemaker import get_execution_role
#session
sess=sagemaker.Session()

#algoritmo de image classifier  que jalaras Aws ECR elastic container registry
algorithm_image=image_uris.retrieve(
    #region de la session
    region=boto3.Session().region_name,
    #algotimo que jalaras, te jalara elmas actualizado
    framework="image-classification"
)

#ubicacion donde se guardara tu modelo entrenado que ocuparas en tu estimator
s3_output_location=f"s3://{bucket}/models/image_model"
print(algorithm_image)


# In[5]:


#lo necesitas para estimator
role=get_execution_role()
print(role)


# In[6]:


import sagemaker
#Estimator object: objeto que llevara las especificaciones (parámetros) para entrenar al modelo
img_classifier_model=sagemaker.estimator.Estimator(
    #algoritmo que se jalo de ECR arriba
    algorithm_image,
    role=role,
    #cuantas instancias se usaran, como tenemos miles solo suaremos una, ademas asi no perdemos accuracy, 
    #si tuvieramos millones, si deberiamos usar mas
    instance_count=1,
    #Instancia con GPU (mas poderosa que cpu) ml.p2.xlarge
    instance_type="ml.p2.xlarge",
    #Gigabytes que se ocupara para guardar el input(training) data del modelo
    volume_size=50,
    #cuantos segundos maximo aguantara el entrenamiento
    max_run=432000,
    #diras que meteras files, si trabajaras con millones de datos deberias usar Pipe
    #al darle file se sagepmaker copiara las imagnes en tu directorio local mientras entrena el modelo
    input_mode="File",
    #donde guardara el modelo
    output_path=s3_output_location,
    #sesion
    sagemaker_session=sess
)
print(img_classifier_model)


# In[7]:


import glob 
count=0
#contar cuantas imagnes hay de train
for filepath in glob.glob('./data/chest_xray/train/*.jpeg'):
    count+=1
print(count)


# In[8]:


count=5216


# In[9]:


#parametros del modelo (ademas de los ya configurados)
img_classifier_model.set_hyperparameters(
image_shape='3,224,224',
    #2 clases porque es binario, normal o pneumonia
num_classes=2,
    #Usaras el modelo pre-netrenadod e amazon, esto es transfer learning
use_pretrained_model=1,
    #numero de pruebas
num_training_samples=count,
    #random augmentations como rotaciones y crops
augmentation_type='crop_color_transform',
    #Epoch: este es el número de veces que se van a pasar cada sample de entrenamiento por la red.
    #lo mas adecuado seria entre 30 y 50 pero queremos ahorrar dinero
epochs=15,
    #si el modelo deja de mejorar su score, va a dejar de entrenarse ahorrando costo
early_stopping=True,
    #cuantos epochs deben pasar minimo para pararse
early_stopping_min_epochs=8,
    #si no mejora mas de 0.0 se detendra
early_stopping_tolerance=0.0,
    #si deja de mejorar, en 5 epochs se detiene
early_stopping_patience=5,
    #learning rate para ir acercandose al minimo, o sea se multiplciara por 0.1 el lr
lr_scheduler_factor=0.1,
    #en que epochs se va a decrementar el learning rate
lr_scheduler_step='8,10,12')


# In[10]:


#categorical parameter: va a elegir uno de los n parametros que le des
#continous: se movera entre los rangos que le des
from sagemaker.tuner import CategoricalParameter,ContinuousParameter,HyperparameterTuner

#diccionario de parametros dinamicos que iran cambiando de valor 
hyperparameter_ranges={
    #el learning rate ira cambiando su valor entre estos rangos
    "learning_rate":ContinuousParameter(0.01,0.1),
    #cuantos samples tomara por iteracion creo que al cambiar de epoch
    "mini_batch_size":CategoricalParameter([8,16,32]),
    #optimizadores para llegar al minimum que tomara
    "optimizer":CategoricalParameter(["sgd","adam"])
}


# In[11]:


#metrica en la que se debe basar el modelo para ajustarse: ACCURACY
objective_metric_name="validation:accuracy"
#Se quiere maximizar el accuracy
objective_type="Maximize"
#Se van a realziar 5 entrenamientos con difenretes hiperparametros (que son los dinamicos de arriba)
max_jobs=5
#cuantos jobs procesar en paralelo, solo uno porque tenemos miles de imagenes, no millones
#es impotante saber tambien que el modelo funciona con bayesian search, que practicamente se va quedando
#con los parametros que den la mejor accuracy con cada entrenamiento
max_parallel_jobs=1


# In[12]:


#en el tuner juntas todos los parametros: estimator, hiperparametro, objetivos y parametros de jobs
tuner=HyperparameterTuner(estimator=img_classifier_model,
                         objective_metric_name=objective_metric_name,
                         hyperparameter_ranges=hyperparameter_ranges,
                         objective_type=objective_type,
                         max_jobs=max_jobs,
                         max_parallel_jobs=max_parallel_jobs  
                         )


# In[13]:


from sagemaker.session import TrainingInput

#ahora le indicas los inputs a tu modelo
model_inputs={
    #Indicas donde estan tus imagenes para entrenar y que son imagenes (,content_type="application/x-image")
    "train":sagemaker.inputs.TrainingInput(s3_data=f"s3://{bucket}/train/",content_type="application/x-image"),
    #validation realemnte es test
    "validation":sagemaker.inputs.TrainingInput(s3_data=f"s3://{bucket}/test/",content_type="application/x-image"),
    #Indicas ubicacion y tipo de contenido de tus archivos lst
    "train_lst":sagemaker.inputs.TrainingInput(s3_data=f"s3://{bucket}/train.lst",content_type="application/x-image"),
    "validation_lst":sagemaker.inputs.TrainingInput(s3_data=f"s3://{bucket}/test.lst",content_type="application/x-image"),
}


# In[14]:


#debes ponerle un nombre unico a cada job (entrenamiento )(seran 5 jobs)
#asi que cada entrenamiento se llevara la hora unica a la que se empezo, por lo que los 5 nombres seran unicos
import time 
job_name_prefix="classifier"
timestamp=time.strftime("-%Y-%m-%d-%H-%M-%S",time.gmtime())
job_name=job_name_prefix+timestamp


# In[ ]:


#entrenas modelo, le pasas inputs, nombre de los trabajos y que los logs se impriman  en cloudwatch
tuner.fit(inputs=model_inputs,job_name=job_name,logs=True)


# In[9]:


#s3://x-ray-pneumonia-ml/models/image_model/classifier-2022-08-11-04-22-19-003-4314f0e6/output/

#nuevamente obtendremos el rol de execution para poder acceder a S3
import sagemaker
from sagemaker import get_execution_role
role=get_execution_role()


# In[12]:


#Modelo final con los mejor hiperparametros tras el entrenamiento
model = sagemaker.model.Model(
    #parametro para seleccionar el algoritmo de imagen de ECR que quieres (el de image classification ajustado de arriba)
image_uri = algorithm_image,
    #los datos del modelo, guardados en s3 (o sea hiperparametros del mejor modelo)
model_data = "s3://x-ray-pneumonia-ml/models/image_model/classifier-2022-08-11-04-22-19-003-4314f0e6/output/model.tar.gz",
role = role
)


# DEPLOYMENT DEL MODELO A UN ENDPOINT DE SAGEMAKER

# In[13]:


#Definicion de endpoints
#asi se llamara tu endpoint que podras encontrar en sagemaker endpoints
endpoint_name='ml-pneumonia-model-endpoint'

#Al modelo se esta haciendo deploy en en un ednpoint AWS sagemaker, o sea que estará guardado en ese endpoint
#y podrás consultarlo desde ahi
deployment=model.deploy(
    #cuantas maquinas usaras para correrlo
initial_instance_count=1,
    #tipo de maquina en el que correra -> cuesta 0.288 USD la hora
instance_type='ml.m4.xlarge',
endpoint_name=endpoint_name)


# In[20]:


from sagemaker.predictor import Predictor
#el predictor jalara el modelo del endpoint
predictor=Predictor("ml-pneumonia-model-endpoint")

from sagemaker.serializers import IdentitySerializer
import base64
#imagen con pneumonia para realizar inferencia
file_name='data/val/val_pneumonia3.jpeg'
#indicandole que leera una imagen
predictor.serializer= IdentitySerializer("image/jpeg")
with open(file_name,"rb")as f:
    #archivo que leera
    payload=f.read()
    
inference=predictor.predict(data=payload)
print("[Probabilidad de NO TENER PNEUMONIA , Probablidad de ENFERMEDAD ]")
print(inference)


# Generar matriz de confusion con las predicciones: vamos a checar cuantos resultados fueron correctos 

# In[21]:


import glob
import json
import numpy as np
#path hacia todas las imagenes
file_path='data/val/*.jpeg'
files=glob.glob(file_path)

y_true=[]
y_pred=[]

def make_pred():
    for file in files:
        #si en la imagen usada de input no hay enfermedad
        if "normal" in file:
            with open(file,"rb") as f:
                #se lee imagen
                payload=f.read()
                #se hace prediccion
                inference=predictor.predict(data=payload).decode("utf-8")
                #se guarda prediccion en un array en lugar de un json creo
                result=json.loads(inference)
                #se toma la posicion del valor maximo del array -> o sea la probabilidad del modelo 
                #recuerda que la inferencia se regresa asi: 
                # [ Posicion0 Probabilidad de NO TENER PNEUMONIA , Posicion 1 Probablidad de ENFERMEDAD ]
                predicted_class=np.argmax(result)
                y_true.append(0)
                y_pred.append(predicted_class)
        elif "pneumonia" in file:
            with open(file,"rb") as f:
                payload=f.read()
                inference=predictor.predict(data=payload).decode("utf-8")
                result=json.loads(inference)
                predicted_class=np.argmax(result)
                y_true.append(1)
                y_pred.append(predicted_class)


# In[22]:


make_pred()
print("Correctos: ", y_true)
print(("Predicciones: ",y_pred)


# La diagonal que va de izq superior a derecha inferior, es la de resultados correctos y la otra diagonal es cuantos estuvieron, mal asi que solo 1 resultado fue incorrecto

# In[23]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_true,y_pred)


# In[24]:


from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))


# In[ ]:




