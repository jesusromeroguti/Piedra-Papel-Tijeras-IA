
from sklearn.neural_network import MLPClassifier
from random import choice

# Opciones que pueden escoger los jugadores
options = ["piedra", "tijeras", "papel"]

################## INICIO Funciones ##################

# Funcion que elige una opcion random de la lista options
def get_choice():
    return choice(options)

# Funcion que encuentra al ganador de la partida
# El resultado 0 equivale a EMPATE
# El resultado 1 equivale a Player1 GANADOR
# El resultado 2 equivale a Player2 GANADOR
def search_winner(player1, player2):
    if player1 == player2:
        result = 0
    elif player1 == "piedra" and player2 == "tijeras":
        result = 1
    elif player1 == "piedra" and player2 == "papel":
        result = 2
    elif player1 == "tijeras" and player2 == "papel":
        result = 1
    elif player1 == "tijeras" and player2 == "piedra":
        result = 2
    elif player1 == "papel" and player2 == "piedra":
        result = 1
    elif player1 == "papel" and player2 == "tijeras":
        result = 2
    return result

# Funcion que convierte los strings (piedra, papel, tijeras) en 0's y 1's para que se pueda procesar la informacion mejor
# Piedra --> [1,0,0]
# Tijeras --> [0,1,0]
# Papel --> [0,0,1]
def str_to_list(option):
    if option == "piedra":
        res = [1,0,0]
    elif option == "tijeras":
        res = [0,1,0]
    else:
        res = [0,0,1]
    return res


''' Funcion que sirve para jugar y a medida que juega vaya 
    aprendiendo sobre las victorias. De esta manera aprendera
    que tijeras gana a papel y que piedra gana a tijeras.'''
def play_and_learn(iters=10, debug=False):
    # Recuento de partidas que hemos ganado y perdido
    score = {"win": 0, "loose": 0}
    
    data_X = []
    data_Y = []
    
    for i in range(iters):
        # Player1 elige una opcion 
        player1 = get_choice()
        
        # Predecimos que opcion escogeria el modelo para ganar la opcion del player1
        # Devolvera una lista de pesos por cada una de las opciones
        # Un porcentage de probabilidad de ganar
        predict = model.predict_proba([str_to_list(player1)])[0]
        
        # Si piedra tiene una probabilidad de ganar de 95%
        if predict[0] >= 0.95:
            player2 = options[0]
        # Si tijeras es la opcion ganadora en un 95%
        elif predict[1] >= 0.95:
            player2 = options[1]
        # Si papel es la opcion ganadora en un 95%
        elif predict[2] >= 0.95:
            player2 = options[2]
        # Si ocurre que no esta seguro en un 95% de ninguna opcion
        else:
            player2 = get_choice()
            
        # Trazas de informacion
        if debug == True:
            print("Player1: %s Predict(modelo): %s --> %s" % (player1, predict, player2))
        
        # Obtenemos el ganador a partir de las elecciones del player1 y el player2
        # Gana el player1 si la salida es 1
        # Gana el player2 si la salida es 2
        # Empate si la salida es 0
        winner = search_winner(player1, player2)
        
        # Trazas de informacion
        if debug == True:
            print("Comprobamos: p1 vs p2 --> %s" % winner)
        
        # Si el player2 gana guardaremos la informacion de la partida para entrenar el modelo con las tiradas ganadoras
        if winner == 2:
            # Metemos en X la opcion del player1
            data_X.append(str_to_list(player1))
            # Metemos en Y la opcion ganadora del player2
            data_Y.append(str_to_list(player2))
            
            # Actualizamos la variable score
            score["win"] +=1
        # Si pierde el player2 
        else:
            score["loose"] +=1
    
    return score, data_X, data_Y


################## FIN Funciones ##################


################## INICIO Programa/MAIN ##################

# Variable que contiene las elecciones del player1
data_X = list(map(str_to_list, ["piedra", "tijeras", "papel"]))
# Variable que contiene las opciones que ganan al player1 que usara la red neuronal para aprender
data_Y = list(map(str_to_list, ["papel", "piedra", "tijeras"]))

print(data_X)
print(data_Y)

# Implementamos la red neuronal con Scikit learn, concretamente el MLPClassifier

clf = MLPClassifier(verbose=False, warm_start=True)

# Modelo
# Solo le vamos a pasar una ventaja porque queremos que a partir 
# de esa aprenda las demas ventajas, si no aprenderia todas de golpe.
model = clf.fit([data_X[0]], [data_Y[0]])

# Vamos a iterar muchas veces y ver como aprende la red
i = 0
# Guardaremos un historico de porcentages victoria
historic_pct = []
while True:
    i+=1
    # Llamamos a la funcion para jugar y que la red aprenda
    # Numero de iteraciones deseadas
    score, data_X, data_Y = play_and_learn(1000, debug=False)
    # Calculamos el porcentage de victorias
    pct = (score["win"]*100/(score["win"]+score["loose"]))
    historic_pct.append(pct)
    print("Iter: %s - score: %s %s %%" % (i, score, pct))
    
    # Si hay algo que aprender, es decir si data_X no esta vacia
    if len(data_X):
        model = model.partial_fit(data_X, data_Y)
    
    # Para que pare el while True
    # Si en el historic_pgt la suma de las 9 partidas has sacado un 100%
    # Paramos porque siempre gana
    if sum(historic_pct[-9:]) == 900:
        break

################## FIN Programa/MAIN ##################