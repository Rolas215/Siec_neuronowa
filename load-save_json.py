# Ładowanie treści z pliku json
# Do dyspozycji są dwie funkcje load (dla pliku json) lub loads (gdy json jest w postaci string'a)
# Taka sama sytuacja jest za zapisaniem do plku w przypadku dump i dumps
from json import load, dump

try:
    with open("nazwa_pliku.json", "r") as input_file:
        content = load(input_file)
        # Wyświetlenie danych jako słownik
        print(content)
        # Wyświetlenie danych za pomocą pętli
        for i in content.get("klucz_z_słownika", []):
            print(i)
except FileNotFoundError:
    print("Nie znaleziono pliku!")
# Dodawanie danych do pliku json
input_data = input()
content["dane"].append({
    "dane": input_data
})
with open("nazwa_pliku.json", "w") as output_file:
    dump(content, output_file)
