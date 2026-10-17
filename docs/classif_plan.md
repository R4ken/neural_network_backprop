# Plan Eksperymentów Klasyfikacyjnych (MLP)

Niniejszy dokument przedstawia szczegółowy plan eksperymentów dla wielowarstwowego perceptronu (MLP) w zadaniu klasyfikacji, opracowany na podstawie wymagań zdefiniowanych w [wymagania.md](file:///c:/Personalne/1_STUDIA/mini_repo/sem7/sieci/p1_mlp/docs/wymagania.md). Plan jest przeznaczony dla kolejnego agenta / programisty implementującego procedury badawcze i wizualizacje.

Zgodnie z wymaganiami:
* Kod źródłowy eksperymentów tworzony jest w języku angielskim.
* Wszelkie generowane artefakty graficzne (wykresy: tytuły, etykiety osi, legendy) **muszą być w języku polskim**.
* Do zbadania włączono dodatkowy, czwarty eksperyment badający wpływ wielkości zbioru danych treningowych (sample complexity & generalization gap).

---

## 1. Zbiory danych i przygotowanie do eksperymentów

W folderze `data/p1/classification` znajdują się dwa zbiory danych w formacie CSV o różnych rozmiarach ($N \in \{100, 500, 1000, 10000\}$):

1. **`data.simple.{train|test}.{N}.csv`**
   * **Struktura wejściowa:** 2 cechy ciągłe $x, y \in \mathbb{R}^2$.
   * **Klasy:** Etykiety $1$ oraz $2$ (klasyfikacja binarna).
   * **Preprocessing:**
     * Mapowanie etykiet: $y \in \{1, 2\} \to \{0, 1\}$ (dla binarnej funkcji celu i wyjścia z 1 neuronem z funkcją sigmoidalną) lub kodowanie one-hot $[1, 0]$ / $[0, 1]$ (dla Softmax z 2 neuronami).
     * Normalizacja / Standaryzacja: Cechy wejściowe są już wycentrowane w okolicach zera (w przybliżeniu $[-1, 1]$), opcjonalnie można zastosować standaryzację z-score wyliczaną wyłącznie na zbiorze treningowym.

2. **`data.three_gauss.{train|test}.{N}.csv`**
   * **Struktura wejściowa:** 2 cechy ciągłe $x, y \in \mathbb{R}^2$ pochodzące z trzech rozkładów gaussowskich.
   * **Klasy:** Etykiety $1, 2, 3$ (klasyfikacja wieloklasowa).
   * **Preprocessing:**
     * Mapowanie etykiet na kodowanie one-hot: klasa $1 \to [1, 0, 0]$, klasa $2 \to [0, 1, 0]$, klasa $3 \to [0, 0, 1]$.
     * Wyjście sieci: 3 neurony z aktywacją `Softmax`.

## 2. Trzy filary wizualizacji procesu uczenia

Zgodnie z wymaganiami projektowymi każdy eksperyment dostarcza 3 komplementarne perspektywy procesu uczenia:

### Filar I: Wizualizacja zbioru uczącego i efektów klasyfikacji
* **Co wizualizujemy:**
  * Przestrzeń cech 2D $(x_1, x_2)$ z wykresem konturowym (`contourf`) prawdopodobieństwa przynależności do klas lub dyskretnych regionów decyzyjnych.
  * Nałożone punkty danych ze zbioru uczącego (przezroczyste kółka) oraz testowego.
  * Opcjonalnie wyróżnienie próbek błędnie zaklasyfikowanych (*misclassified samples*), co ułatwia analizę trudnych obszarów brzegowych.

### Filar II: Wizualizacja błędu propagowanego w kolejnych iteracjach uczenia
* **Co wizualizujemy:**
  * **W skali iteracji (kroków mini-batch):** Wartość błędu (loss) po każdym kroku optymalizatora SGD. Pozwala to zaobserwować stochastyczny szum gradientu oraz dynamikę propagacji błędu na samym początku treningu (np. pierwsze kilkaset iteracji).
  * **W skali epok:** Średnia strata na zbiorze uczącym (`loss`) oraz na zbiorze walidacyjnym/testowym (`val_loss`), uzupełniona o krzywą dokładności (`accuracy` i `val_accuracy`).
  * Wykresy w języku polskim z wyraźną legendą: *Zbiór uczący*, *Zbiór testowy*.

### Filar III: Wizualizacja wartości wag w kolejnych iteracjach uczenia
* **Co wizualizujemy:**
  * **Trajektorie wybranych konkretnych wag:** Śledzenie wartości 4–6 wybranych parametrów $w_{ij}(t)$ z warstwy wejściowej i wyjściowej w kolejnych epokach/iteracjach, pokazujące jak wagi ewoluują w czasie i kiedy osiągają plateau.
  * **Normy Frobeniusa macierzy wag:** $\|W^{(l)}\|_F = \sqrt{\sum w_{ij}^2}$ dla każdej warstwy $l$ w funkcji epok. Pozwala to natychmiast wychwycić zjawisko zanikającego gradientu (brak wzrostu normy w początkowych warstwach) lub eksplozji wag.
  * **Rozkład wag przed i po treningu:** Histogramy wag dla warstw ukrytych i wyjściowej na początku (po inicjalizacji) i na końcu procesu uczenia.

---

## 3. Wymagane poprawki i rozszerzenia w bibliotece `src/py_nn1`

Przed uruchomieniem eksperymentów należy zaadresować następujące kwestie w istniejącej implementacji:

1. **Błąd w pętli mini-batch (`src/py_nn1/utils.py`):**
   * Funkcja `batch_iterator` zawiera obecnie zdublowaną pętlę po mini-batchach (generuje $2\times$ za dużo batchy na epokę). Należy usunąć redundantny drugi blok `for start in range(...)`.
2. **Śledzenie historii wag i iteracji w `MLP` (`src/py_nn1/mlp.py`):**
   * Zgodnie z wymaganiem *"wizualizacja wartości wag w kolejnych iteracjach uczenia"*, klasa `MLP` (lub callback/historia) powinna rejestrować:
     * Wartości wag co epokę (lub co zadaną liczbę iteracji): normy $\|W\|_F$ dla każdej warstwy oraz wartości próbki wybranych pojedynczych współczynników wagowych.
     * Historię błędu na poziomie poszczególnych iteracji (`iteration_loss`) oraz epok (`epoch_loss`, `val_loss`).
3. **Śledzenie metryki dokładności (Accuracy):**
   * `fit()` rejestruje obecnie tylko `loss` i `val_loss`. Należy rozszerzyć rejestrację o `accuracy` i `val_accuracy` w każdej epoce.
4. **Zarządzanie ziarnem losowości (`set_seed`):**
   * Każdy eksperyment musi umożliwiać powtarzalność wyników (np. seed = 42). Dla rzetelności porównań zaleca się wykonanie każdego wariantu na 3–5 ziarnach losowych i raportowanie wartości średniej oraz odchylenia standardowego.
5. **Zależność biblioteczna:**
   * Dodać `matplotlib` do środowiska (`uv add matplotlib`), aby umożliwić renderowanie wykresów do plików `.png`.

---

## 4. Szczegółowy plan eksperymentów

---

### Eksperyment 1: Wpływ funkcji aktywacji w warstwach ukrytych

#### 1. Cel i hipoteza
Zbadanie wpływu wyboru funkcji aktywacji w warstwach ukrytych na dynamikę uczenia, problem zanikającego gradientu, prędkość zbieżności oraz kształt uzyskanych granic decyzyjnych.
* **Hipoteza:** ReLU osiągnie najszybszą zbieżność i uniknie nasycenia gradientów. Sigmoida w warstwach ukrytych będzie uczyć się wolniej z powodu zanikających gradientów. Tanh, będąc funkcją symetryczną o środku w zerze, sprawdzi się lepiej niż Sigmoida, ale może ustępować ReLU przy głębszych konfiguracjach.

#### 2. Dane wejściowe i podział
* **Zbiory danych:**
  * `data.simple.train.1000.csv` oraz ewaluacja na `data.simple.test.1000.csv`
  * `data.three_gauss.train.1000.csv` oraz ewaluacja na `data.three_gauss.test.1000.csv`
* Zbiór treningowy można podzielić na część uczącą (800 próbek) i walidacyjną (200 próbek) lub traktować test set jako ewaluator końcowy.

#### 3. Konfiguracja hiperparametrów
* **Porównywane funkcje aktywacji w warstwach ukrytych:**
  1. `Sigmoid`
  2. `ReLU`
  3. `Tanh`
* **Warstwa wyjściowa:**
  * Dla `data.simple`: 1 neuron z `Sigmoid` (i `BinaryCrossEntropyLoss`) lub 2 neurony z `Softmax` (i `CategoricalCrossEntropyLoss`).
  * Dla `data.three_gauss`: 3 neurony z `Softmax` (i `CategoricalCrossEntropyLoss`).
* **Architektura bazowa:** Stała, np. 2 warstwy ukryte: `[2, 32, 16, C]`.
* **Pozostałe parametry stałe:**
  * Optymalizator: `SGD(learning_rate=0.05)`
  * Rozmiar paczki: `batch_size = 32`
  * Liczba epok: `epochs = 150`
  * Inicjalizacja wag: `xavier` dla Sigmoid/Tanh, `he` dla ReLU (oraz opcjonalnie `xavier` dla wszystkich dla zachowania identycznych warunków startowych).

#### 4. Wytwarzane wykresy (w języku polskim)
1. **Przebieg błędu i dokładności w czasie:**
   * Wykres liniowy: *„Funkcja straty w kolejnych epokach dla różnych funkcji aktywacji”* (osie: Epoka vs Wartość funkcji straty; osobne krzywe dla Sigmoid, ReLU, Tanh – zbiór uczący i testowy).
   * Wykres liniowy: *„Dokładność klasyfikacji (Accuracy) w kolejnych epokach”* (osie: Epoka vs Dokładność [%]).
2. **Granice decyzyjne:**
   * Zestawienie 3 podwykresów (subplotów): *„Granice decyzyjne w przestrzeni cech (Sigmoid vs ReLU vs Tanh)”* z nałożonymi punktami testowymi.
3. **Ewolucja wag:**
   * Wykres liniowy: *„Ewolucja normy wag $\|W\|$ w kolejnych epokach uczenia”* dla każdej warstwy przy poszczególnych aktywacjach (uwidocznienie tempa aktualizacji wag i zanikania gradientów).

---

### Eksperyment 2: Wpływ liczby warstw ukrytych oraz ich liczności

#### 1. Cel i hipoteza
Analiza zdolności aproksymacyjnej sieci perceptronowej w zależności od jej głębokości (od 0 do 4 warstw ukrytych) i szerokości (liczby neuronów).
* **Hipoteza:**
  * 0 warstw ukrytych (klasyfikator liniowy) nie będzie w stanie poprawnie rozdzielić danych o nieliniowych granicach (szczególnie `three_gauss` lub nieliniowych wygięć w `simple`).
  * 1–2 warstwy ukryte o umiarkowanej szerokości pozwolą na elastyczne dopasowanie gładkich granic decyzyjnych bez przeuczenia.
  * 3–4 warstwy ukryte mogą wykazywać wolniejszą zbieżność w prostym SGD (potrzeba precyzyjniejszego strojenia współczynnika uczenia) i tendencję do overfitingu na małych zbiorach.

#### 2. Dane wejściowe i podział
* `data.simple.train.1000.csv` / `data.simple.test.1000.csv`
* `data.three_gauss.train.1000.csv` / `data.three_gauss.test.1000.csv`

#### 3. Badane architektury
Należy przetestować warianty od 0 do 4 warstw ukrytych:
1. **0 warstw ukrytych:** `[2, C]` (model liniowy / regresja logistyczna).
2. **1 warstwa ukryta (porównanie szerokości):**
   * Wąska: `[2, 4, C]`
   * Standardowa: `[2, 16, C]`
   * Szeroka: `[2, 64, C]`
3. **2 warstwy ukryte:**
   * `[2, 16, 8, C]`
   * `[2, 32, 16, C]`
4. **3 warstwy ukryte:**
   * `[2, 32, 16, 8, C]`
5. **4 warstwy ukryte:**
   * `[2, 64, 32, 16, 8, C]`

* **Parametry stałe:**
  * Funkcja aktywacji w warstwach ukrytych: wybrana najlepsza z Eksperymentu 1 (np. `ReLU` lub `Tanh`).
  * Funkcja aktywacji na wyjściu: `Softmax` (dla multi-class) / `Sigmoid` (dla binary).
  * Funkcja straty: `CrossEntropy`.
  * Optymalizator: `SGD(learning_rate=0.03)`, `batch_size=32`, `epochs=150`.

#### 4. Wytwarzane wykresy (w języku polskim)
1. **Zestawienie skuteczności w zależności od liczby warstw:**
   * Wykres słupkowy / punktowy: *„Wpływ liczby warstw ukrytych na dokładność klasyfikacji na zbiorze testowym”* (oś X: Architektura, oś Y: Dokładność [%]).
2. **Krzywe uczenia dla wybranych architektur:**
   * Wykres liniowy: *„Przebieg straty w funkcji epok dla architektur o głębokości od 0 do 4 warstw”*.
3. **Porównanie granic decyzyjnych:**
   * Siatka wykresów (grid 2x3 lub 1x4): *„Wpływ głębokości architektury na kształt granic decyzyjnych”* (widoczne przejście z podziału liniowego dla 0 warstw do złożonych nieliniowych regionów decyzyjnych).
4. **Ewolucja wag w sieci głębokiej:**
   * Wykres liniowy: *„Ewolucja norm wag poszczególnych warstw w sieci 4-warstwowej”* (porównanie warstwy wejściowej vs wyjściowej).

---

### Eksperyment 3: Wpływ miary błędu na wyjściu sieci na skuteczność uczenia

#### 1. Cel i hipoteza
Porównanie skuteczności optymalizacji funkcji błędu średniokwadratowego (`MSELoss`) oraz entropii krzyżowej (`BinaryCrossEntropyLoss` / `CategoricalCrossEntropyLoss`) w zadaniu klasyfikacji.
* **Hipoteza:** Entropia krzyżowa jest naturalną funkcją straty dla modeli probabilistycznych (związana z metodą największej wiarygodności). W połączeniu z funkcjami aktywacji `Sigmoid`/`Softmax`, błąd MSE powoduje problem tzw. nasycenia gradientu (gradient saturation) – gdy sieć generuje błędne przewidywanie z dużym prawdopodobieństwem, pochodna sigmoidy dąży do 0, co drastycznie spowalnia proces uczenia. Entropia krzyżowa eliminuje ten problem, zapewniając silniejszy gradient i szybszą zbieżność.

#### 2. Dane wejściowe i podział
* `data.simple` (klasyfikacja binarna):
  * Wyjście: 1 neuron (`Sigmoid`), porównanie `BinaryCrossEntropyLoss` vs `MSELoss`.
* `data.three_gauss` (klasyfikacja wieloklasowa):
  * Wyjście: 3 neurony (`Softmax`), porównanie `CategoricalCrossEntropyLoss` vs `MSELoss` (z kodowaniem one-hot).

#### 3. Konfiguracja hiperparametrów
* Architektura: stała, optymalna z poprzednich eksperymentów (np. `[2, 32, 16, C]`, aktywacja `ReLU` w warstwach ukrytych).
* Identyczne ziarno inicjalizacji losowej (`set_seed`) dla obu miar błędu.
* Testowane wartości learning rate (np. $lr \in \{0.01, 0.05, 0.1\}$), aby zweryfikować czy MSE wymaga innego kroku uczenia.
* `epochs = 150`, `batch_size = 32`.

#### 4. Wytwarzane wykresy (w języku polskim)
1. **Krzywe uczenia:**
   * Wykres liniowy: *„Porównanie zbieżności: Entropia Krzyżowa vs Błąd Średniokwadratowy (MSE)”* z podwójną osią Y lub znormalizowaną dokładnością:
     * Osobny wykres dokładności (Accuracy) w czasie dla obu funkcji błędu.
2. **Wpływ funkcji straty na granice decyzyjne:**
   * Wykresy 2D: *„Granice decyzyjne wyznaczone przy optymalizacji Cross-Entropy oraz MSE”*.
3. **Ewolucja wag i gradientów:**
   * Wykres liniowy: *„Dynamika zmian wag ($\Delta W$) w początkowych epokach uczenia: Cross-Entropy vs MSE”* (pokazujący tempo ucieczki z punktów siodłowych / nasycenia).

---