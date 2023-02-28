import json
import boto3
#para formatizar imagenes
import base64

#lo habiamos borrado pero al final lo vovleremos a crear
endpoint_name = "ml-pneumonia-model-endpoint"

#pa acceder a sagemaker
sagemaker_runtime_client = boto3.client('runtime.sagemaker')

#todas las funciones de lambda deben tener un lambda handler
#el evento sera lo que recibe la funcion lambda desde la API, 

#en este caso es una imagen codificada en base64 que es la manera mas optima de mandar imagenes
def lambda_handler(event, context):
    #este print se vera en cloudwatch
    print(event)
    #imagen decodificada que se tomara como input para la prediccion
    image = base64.decoded(event['image'])
    #se vera en cloudwatch
    print(image)
    
    #se llama a la funcion de prediccion con la imagen formatizada
    return _predictPneumonia(image)

#esta es la que invocara al endpoint
def _predictPneumonia(image):
    #respuesta al invocar endpoint
    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName = endpoint_name,
        #para checar que se reciba una imagen
        ContentType = "application/x-image",
        #el body de la respuesta sera la imagen y creo que aqui se guardara de regreso el resultado de la inferwncia
        Body = image
        )
    
    #vas a leer y guardar la respuesta de la imagen
    result = response['Body'].read()
    #en la linea de arriba result es un array de bytes asi que lo haras un array normal
    result = json.loads(result)
    print("result ", result)
    #recuerda que el formato del resultado es [ Posicion0 Probabilidad de NO TENER PNEUMONIA , Posicion 1 Probablidad de ENFERMEDAD ]
    #entonces el valor de la posicion 0 es mayor, se guarda un 0 porque quiere que hay mas probabilidad de que no este enfermo
    #y si la prodabilidad de estar enfermo es mayor se guarda un 1
    #0: sano , 1: pneumonia
    predicted_class = 0 if result[0] > result[1] else 1
    #de la misma manera que arriba enviaremos tambien la probabilidad -> if en una sola linea dinamico
    probability = result[0] if result[0] > result[1] else result[1]
    
    #devuelves el resultado
    if predicted_class == 0:
        #yo digo que hace falta cerrar comilla
        return f"NO PNEUMONIA - probability of : {probability}"
    else:
        return f"PNEUMONIA DETECTED - probability of : {probability}"
    