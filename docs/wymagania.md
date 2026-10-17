# 2.1. Część pierwsza (15 punktów)
Program pozwalający rozwiązywać zadania klasyfikacji i regresji z użyciem perceptronu wielowarstwowego i wizualizujący jego działanie, w tym uczenie sieci z użyciem algorytmu propagacji wstecznej błędu (SGD lub jego warianty). Wymagana są wizualizacje:
- wyników uczenia sieci dla problemu regresji dla funkcji R → R i problemu klasyfikacji wektorów w R2;
- przebieg błędu po iteracjach;
- ewolucja wartości wag.
## 2.1.1. Zbiory danych
Dane dostępne są w folderze ```data```. Podzielono je na zadania ```regression``` oraz ```classification```.
Pierwszy wiersz każdego pliku csv zaczyna się opisem kolumn.

## 2.1.2. Uwagi do implementacji
Implementację należy wykonać na poziomie podstawowym, tak aby kod pokazywał zrozumienie zasad działania i uczenia sieci. Nie można korzystać z bibliotek implementujących sieci neuronowe. Można korzystać z biblioteki numpy.
Implementacja jest obecna w folderze ```src/py_nn1/```.
Eksperymenty powinny zostać zwieńczone wykresami z poprawnie opisanymi osiami, legendą i tytułami.
Kod jest w języku angielskim, ale artefakty w postaci wykresów powinny być w języku polskim.

## 2.1.3. Elementy do zaimplementowania
- możliwość zainicjowania (powtarzalnego) procesu uczenia z zadanym ziarnem generatora liczb losowych
- łatwa konfiguracja liczby warstw w sieci i neuronów w warstwie, obecności biasów
- wizualizacja zbioru uczącego i efektów klasyfikacji oraz regresji
- wizualizacja błędu propagowanego w kolejnych iteracjach uczenia
- wizualizacja wartości wag w kolejnych iteracjach uczenia

## 2.1.4. Elementy do zbadania
- Wpływ funkcji aktywacji na skuteczność działania sieci – sprawdzić funkcję sigmoidalną i dwie inne wybrane (funkcje aktywacji wewnątrz sieci). Uwaga: funkcja aktywacji na wyjściu musi być dobrana odpowiednio do rodzaju problemu.
- Wpływ liczby warstw ukrytych w sieci oraz ich liczności. Zbadać różne liczby warstw od 0 do 4, kilka różnych architektur.
- Wpływ miary błędu na wyjściu sieci na skuteczność uczenia. Sprawdzić dwie miary błędu dla klasyfikacji i dwie dla regresji.
- Wyniki i wnioski należy koniecznie opisać w raporcie z projektu.

