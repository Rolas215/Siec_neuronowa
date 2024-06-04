import numpy as np
import configparser
import argparse
import json

# Blokuje wyświetlania liczb w notacji naukowej dla biblioteki numpy
np.set_printoptions(suppress=True)


# Wstępna domyślna inicjalizacja zmiennych
Import_file = 'zmienne.json'
Export_file = 'zmienne.json'
n = 500
Layers_init = {"1": [2, 4],
               "2": [4, 1]}

# Wczytanie konfiguracji
config = configparser.ConfigParser()
try:
    config.read('config.ini')
    Import_file = config['DEFAULT']['Import_file']
    Export_file = config['DEFAULT']['Export_file']
    n = int(config['DEFAULT']['Repeat_n'])
    Layers_init = json.loads(config['DEFAULT']['Layers_init'])
except KeyError:
    print("Nie znaleziono poprawnego pliku konfiguracyjnego!")


# Inicjalizacja parsera [argparse]
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--import-file', nargs='?', const="NONE",
                    help='arg="Nazwa pliku" OR None | importuje plik')
parser.add_argument('-e', '--export-file', nargs='?', const="NONE",
                    help='arg="Nazwa pliku" OR None | eksportuje plik')
parser.add_argument('-n', '--num', type=int, nargs='?', default=-1,
                    help='arg=liczba OR None | ustawia liczbę powótrzeń uczenia sieci')
parser.add_argument('-l', '--layers-init', default="NONE",
                    help='arg=\'{\\"1\\": ["liczba wejść", "liczba wyjść"], \\"2\\": [], itd.}\' OR None | ustawia inicjalizację warstw sieci chronologicznie do podanych wartości')
parser.add_argument('-c', '--config-save', action='store_true',
                    help='arg=None | zapisuje podane wartości parsera do konfiguracji nie wykonując programu')
args = parser.parse_args()

# Zapis zmiennych z parsera [argparse]
if(args.import_file and args.import_file != "NONE"):
    Import_file = args.import_file
if(args.export_file and args.export_file != "NONE"):
    Export_file = args.export_file
if(args.num and args.num >= 0):
    n = args.num
if(args.layers_init and args.layers_init != "NONE"):
    Layers_init = json.loads(args.layers_init)


# Zapisanie konfiguracji
if(args.config_save):
    config['DEFAULT'] = {'Import_file': Import_file,
                        'Export_file': Export_file,
                        'Repeat_n': n,
                        'Layers_init': json.dumps(Layers_init)}
    with open('config.ini', 'w') as configfile:
        config.write(configfile)



def sigmoid(inputs):
    """
    Oblicza funkcję sigmoidalną dla danego wejścia.

    Funkcja sigmoidalna jest zdefiniowana jako:
    σ(x) = 1 / (1 + e^(-x))

    Parametry:
    inputs (numpy.ndarray lub float): Wartość lub tablica wartości, dla których ma być obliczona funkcja sigmoidalna.

    Zwraca:
    numpy.ndarray lub float: Wartość lub tablica wartości po zastosowaniu funkcji sigmoidalnej.
    """
    return 1/(1+np.exp(np.negative(inputs)))

def d_sigmoid(inputs):
    """
    Oblicza pochodną funkcji sigmoidalnej dla danego wejścia.

    Pochodna funkcji sigmoidalnej jest zdefiniowana jako:
    σ'(x) = σ(x) * (1 - σ(x))
    gdzie σ(x) jest wartością funkcji sigmoidalnej dla x.

    Parametry:
    inputs (numpy.ndarray lub float): Wartość lub tablica wartości, dla których ma być obliczona pochodna funkcji sigmoidalnej.

    Zwraca:
    numpy.ndarray lub float: Wartość lub tablica wartości po zastosowaniu pochodnej funkcji sigmoidalnej.
    """
    sigmoidP = sigmoid(inputs)
    return sigmoidP*(1-sigmoidP)



# Warstwa
class Layer:
    def __init__(self, input_n, output_n):
        self.weights = [[-2]*output_n]*input_n #np.random.rand(input_n, output_n)*2-2
        self.biases = [0]*output_n
        self.Y = 0
        self.X = 0

    def forward(self, inputs):
        self.X = inputs
        self.Y = np.dot(self.X, self.weights) + self.biases

# Sieć neuronowa
class Neural:
    """
    Neural Network Class
    
    Parameters
    ----------
        `layers`
            Lista utworzonych warstw sieci neuronowej, podana w kolejności chronologicznej
    """
    def __init__(self, layers):
        self.layers = layers
        if (type(self.layers) != list):
            self.layers = [self.layers]

    def test(self, input):
        self.layers[0].forward(input)

        for j in range(1, len(self.layers)):
            self.layers[j].forward(sigmoid(self.layers[j-1].Y))

    def get_net_output(self):
        return sigmoid(self.layers[-1].Y)
    
    def train(self, input, output):
        self.test(input)
        self.Backprop(d_sigmoid, output)
        
    def Backprop(self, derivative_func, y_true):
        ratio = (2*(y_true-sigmoid(self.layers[-1].Y))).T
        for layer in reversed(self.layers):
            ratio = derivative_func(layer.Y)*ratio.T

            layer.weights += np.dot(np.array(layer.X).T, ratio)
            layer.biases += sum(ratio)

            ratio = np.dot(layer.weights, ratio.T)
    
    def data_import(self, file):
        try:
            with open(file, "r") as input_file:
                content = json.load(input_file)
                for i in range(len(self.layers)):
                    self.layers[i].weights = list(list(content.values())[i].values())[0]
                    self.layers[i].biases = list(list(content.values())[i].values())[1]
        except FileNotFoundError:
            print("Nie znaleziono pliku do zaimportowania!")

    def data_export(self, file):
        n = 1
        data_to_save = {}
        for layer in self.layers:
            data_to_save[f'layer{n}'] = {
                'weights': layer.weights.tolist(),
                'biases': layer.biases.tolist()
            }
            n += 1
        with open(file, 'w') as json_file:
            json.dump(data_to_save, json_file, indent=4)


if(not args.config_save):
    # Dane do uczenia sieci
    X = [[1,1],[1,0],[0,1],[0,0]]
    Y = [[1],[0],[0],[0]]

    # Inicjalizacja sieci
    Layers = [Layer(i[0], i[1]) for i in list(Layers_init.values())]
    net = Neural(Layers)

    # Import
    if(args.import_file):
        try:
            net.data_import(Import_file)
        except:
            print("Błędnie skonfigurowana sieć!")

    # Uczenie sieci
    for i in range(n):
        net.train(X, Y)

    # Export
    if(args.export_file):
        net.data_export(Export_file)

    try:
        with open("historical-results.json", "r") as input_file:
            content = json.load(input_file)
    except FileNotFoundError:
        content = {}

    # Ensure 'data' key exists in the content dictionary
    if 'data' not in content:
        content['data'] = []

    # Test Network
    net.test(X)
    print(net.get_net_output())

    data_results = {
        'results': net.get_net_output().tolist()
    }

    content["data"].append(data_results)

    try:
        with open("historical-results.json", "w") as output_file:
            json.dump(content, output_file, indent=4)
    except FileNotFoundError:
        print("Nie znaleziono pliku!")
