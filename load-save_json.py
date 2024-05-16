# Ładowanie treści z pliku json
# Do dyspozycji są dwie funkcje load (dla pliku json) lub loads (gdy json jest w postaci string'a)
# Taka sama sytuacja jest za zapisaniem do plku w przypadku dump i dumps
import json


dictionary = {1: 2132,
              2: 3211}

with open("zmienne.json", "w") as json_file:
    json.dump(dictionary, json_file)


with open("zmienne.json", "r") as input_file:
    content = json.load(input_file)
    # Wyświetlenie danych jako słownik
    print(content)
    # Wyświetlenie danych za pomocą pętli
    for i in content.get("klucz_z_słownika", []):
        print(i)
