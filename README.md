# NLP_tripmodel
Basicamente es lo mismo que la tarea anterior.  En este caso corri dos veces el modelo, la primera vez con un comentario positivo y otro con un comentario negativo. Describo mi interpretacion en cada uno de ellos.

Para entrenar el modelo y debemos ejecutar estos comandos

from src.models.train import * 

from src.models.predict import *

var1  = load_doc()

lda_model(var1) 

Estas instrucciones entrenan el modelo en base a los reviews, y esciben en un pickle el resultado del modelo.  Esto con el objetivo de cargarlo despues y poder ejecutar el predict, que se da con las siguientes intrucciones.

var2 = load_model()

test(var2, 'I really like the hotel, it was a very nice experience, the bedroom was really nice')

test(var2, 'I really like the hotel, it was a very nice experience, the bedroom was really nice')
(4, '0.043*"stay" + 0.028*"place" + 0.022*"good" + 0.022*"staff" + 0.021*"nice" + 0.018*"go" + 0.017*"would" + 0.016*"get" + 0.016*"clean" + 0.014*"also"')

Como mencione anteriormente, aca se corre con un comentario positivo, y lo que logro interpretar que da el modelo da las palabras 'stay', 'place' y 'good' asumo que fue una experiencia agradable y es un buen lugar para quedarse.

test(var2, 'My visit was a disaster, I wont recommend to stay there, you should go to another place')
(4, '0.043*"stay" + 0.028*"place" + 0.022*"good" + 0.022*"staff" + 0.021*"nice" + 0.018*"go" + 0.017*"would" + 0.016*"get" + 0.016*"clean" + 0.014*"also"')

Asumo que si menciona la palabra clean, de lo que estan hablando es que no es un lugar muy limpio.
