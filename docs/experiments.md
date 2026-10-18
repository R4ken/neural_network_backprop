# Zunifikowany Plan Eksperymentów: Klasyfikacja i Regresja (MLP)

Niniejszy dokument stanowi zunifikowany plan badawczy dla wielowarstwowego perceptronu (MLP) realizujący zadania **klasyfikacji** oraz **regresji**, opracowany ściśle na podstawie formalnych wytycznych z [wymagania.md](file:///c:/Personalne/1_STUDIA/mini_repo/sem7/sieci/p1_mlp/docs/wymagania.md).

Plan został zaprojektowany modularnie: wspólne założenia metodologiczne, zasady wizualizacji oraz wymagania architektoniczne opisano jednokrotnie, a szczegółowe procedury badawcze podzielono na dedykowane sekcje dla regresji i klasyfikacji.

---

## 1. Założenia formalne i metodyka badawcza

1. **Język implementacji i wizualizacji:**
   * Całość kodu źródłowego, modułów badawczych i skryptów eksperymentalnych tworzona jest w **języku angielskim**.
   * Wszelkie generowane artefakty wynikowe (wykresy: tytuły, etykiety osi $X$ i $Y$, legendy, adnotacje) **muszą być sporządzone w języku polskim**.
2. **Powtarzalność i rzetelność statystyczna:**
   * Każdy eksperyment wykorzystuje jawnie zainicjowany generator liczb losowych (`set_seed`).
   * Dla wyeliminowania losowości inicjalizacji wag zaleca się powtórzenie kluczowych przebiegów dla 3–5 ziaren (np. `seed ∈ {42, 123, 999}`) i raportowanie wartości średniej oraz odchylenia standardowego ($\mu \pm \sigma$).
3. **Czysta implementacja (zgodnie z 2.1.2):**
   * Wykorzystanie wyłącznie własnej biblioteki `src/py_nn1/` opartej o `NumPy`. Zakaz stosowania gotowych bibliotek sieci neuronowych (PyTorch, TensorFlow, Scikit-learn).
4. **Zarządzanie biasem:**
   * Zgodnie z wymaganiem łatwej konfiguracji obecności biasów, domyślnie eksperymenty przeprowadzane są z włączonym wektorem przesunięcia (`use_bias=True`).

---

## 2. Zbiory danych i specyfika zadań

Wszystkie dane znajdują się w katalogu `data/p1/` i posiadają warianty o liczności $N \in \{100, 500, 1000, 10000\}$ próbek z podziałem na podzbiory uczące (`train`) i testowe (`test`).

```
data/p1/
├── classification/
│   ├── data.simple.{train|test}.{N}.csv
│   └── data.three_gauss.{train|test}.{N}.csv
└── regression/
    ├── data.activation.{train|test}.{N}.csv
    └── data.cube.{train|test}.{N}.csv
```

### 2.1. Zadania regresji ($\mathbb{R} \to \mathbb{R}$)
* **`data.activation`**:
  * Wejście: 1 cecha ciągła $x \in [-2, 5]$.
  * Wyjście: 1 wartość ciągła $y \in [-265, -2]$ (funkcja o nieliniowym przebiegu sigmoidalnym/przejściowym z szerokim zakresem wartości).
* **`data.cube`**:
  * Wejście: 1 cecha ciągła $x \in [-2, 5]$.
  * Wyjście: 1 wartość ciągła $y \in [-198, 86]$ (funkcja wielomianowa stopnia 3 z szumem).
* **Specyfika wyjścia sieci i preprocessingu:**
  * **Warstwa wyjściowa:** 1 neuron liniowy (`Dense(..., out_features=1)`) **bez nieliniowej funkcji aktywacji** (aktywacja tożsamościowa), co pozwala na generowanie wartości w dowolnym przedziale $\mathbb{R}$.
  * **Standaryzacja / Skalowanie:** Ze względu na duże wartości celu $y$, zaleca się z-score standaryzację cech $x$ oraz celów $y$ (parametry liczone wyłącznie na `train`, aplikowane na `test`) lub odpowiednio mały współczynnik uczenia (`learning_rate`), aby zapobiec eksplozji gradientu w początkowych krokach SGD.

### 2.2. Zadania klasyfikacji ($\mathbb{R}^2 \to \mathcal{C}$)
* **`data.simple` (Klasyfikacja binarna):**
  * Wejście: 2 cechy ciągłe $x_1, x_2 \in \mathbb{R}^2$.
  * Etykiety: $y \in \{1, 2\} \to$ mapowane na $\{0, 1\}$.
  * Warstwa wyjściowa: 1 neuron z aktywacją `Sigmoid` (lub 2 neurony z `Softmax`).
* **`data.three_gauss` (Klasyfikacja wieloklasowa):**
  * Wejście: 2 cechy ciągłe $x_1, x_2 \in \mathbb{R}^2$ z trzech skupień gaussowskich.
  * Etykiety: $y \in \{1, 2, 3\} \to$ kodowanie one-hot $\mathbb{R}^3$.
  * Warstwa wyjściowa: 3 neurony z aktywacją `Softmax`.

---

## 3. Zunifikowane standardy wizualizacji (Trzy filary)

Zgodnie z punktami 2.1 oraz 2.1.3 wymagań, każdy badany wariant modelu analizowany jest w oparciu o trzy spójne filary wizualizacyjne:

```
+-----------------------------------------------------------------------------------------+
|                                TRZY FILARY WIZUALIZACJI                                 |
+------------------------------------+----------------------------------+-----------------+
| FILAR I: Zbiór i efekty działania  | FILAR II: Przebieg błędu         | FILAR III: Wagi |
| - Regresja: krzywa aproksymacji 1D | - Skala iteracji (kroki batch)   | - Normy ||W||_F |
| - Klasyfikacja: granice 2D         | - Skala epok (train vs test)     | - Ścieżki w_ij  |
| - Punkty uczące vs testowe         | - Metryki (Loss, Acc, MAE, R^2)  | - Histogramy    |
+------------------------------------+----------------------------------+-----------------+
```

### Filar I: Wizualizacja zbioru danych i efektów działania sieci
* **Dla zadań regresji ($\mathbb{R} \to \mathbb{R}$):**
  * Wykres rozrzutu próbek zbioru uczącego (przezroczyste punkty) oraz testowego.
  * Ciągła linia aproksymacji modelu $\hat{y} = f(x)$, wyliczona dla gęstej, posortowanej siatki argumentów $x \in [x_{\min}, x_{\max}]$.
  * Widoczna zdolność modelu do interpolacji oraz gładkość wyznaczonej funkcji.
* **Dla zadań klasyfikacji ($\mathbb{R}^2 \to \mathcal{C}$):**
  * Wykres konturowy (`contourf`) przestrzeni decyzyjnej lub mapy prawdopodobieństwa wygenerowany na gęstej siatce 2D $(x_1, x_2)$.
  * Nałożone punkty danych ze zbioru uczącego i testowego, pokolorowane wg klas.
  * Opcjonalne wyróżnienie próbek błędnie zaklasyfikowanych.

### Filar II: Wizualizacja błędu propagowanego w kolejnych iteracjach uczenia
* **Skala iteracji (kroków mini-batch):**
  * Rejestracja straty po każdym uaktualnieniu wag przez SGD (szczególnie pierwsze 200–500 iteracji), obrazująca szum stochastyczny i natychmiastową dynamikę zbieżności.
* **Skala epok:**
  * Wykres liniowy porównujący błąd na zbiorze uczącym (`loss`) i testowym (`val_loss`) w kolejnych epokach.
  * Równoległy wykres metryki jakości:
    * Dla klasyfikacji: Dokładność (`accuracy` oraz `val_accuracy` w %).
    * Dla regresji: Średni błąd bezwzględny (`MAE`) lub współczynnik determinacji ($R^2$).
  * Wyraźne oznaczenia osi i legendy w języku polskim: *Krok optymalizacji / Epoka*, *Funkcja straty*, *Zbiór uczący*, *Zbiór testowy*.

### Filar III: Wizualizacja wartości wag w kolejnych iteracjach uczenia
* **Normy Frobeniusa macierzy wag:**
  * Przebieg normy $\|W^{(l)}\|_F = \sqrt{\sum_{i,j} (w_{ij}^{(l)})^2}$ dla każdej warstwy $l$ w funkcji epok. Pozwala na jednoznaczną identyfikację zjawiska zablokowania uczenia, zanikającego gradientu (brak przyrostu normy) lub eksplozji wag.
* **Trajektorie wagi i biasu pierwszego neuronu (Standard 1x3 dla reprezentatywnego ziarna):**
  * Zunifikowany wykres wielopanelowy 1x3 prezentujący ewolucję pierwszej wagi wejściowej $w_{0,0}$ (linia ciągła) oraz biasu $b_0$ (linia przerywana) pierwszego neuronu dla każdej kolejnej warstwy w funkcji epok dla **reprezentatywnego modelu** (pierwsze ziarno losowe, np. `seed = 42`). 
  * *Uzasadnienie metodologiczne:* Ze względu na symetrię permutacji neuronów ukrytych (*permutation symmetry*), ten sam indeks neuronu w różnych ziarnach pełni inną rolę w sieci. Uśrednianie wag pojedynczych neuronów po wielu ziarnach prowadzi do sztucznego znoszenia się wartości do zera i myląco dużego odchylenia standardowego. Dlatego wielkości niezmiennicze na permutacje (Loss, Accuracy, $\|W\|_F$) uśredniamy po ziarnach ($\mu \pm \sigma$), natomiast trajektorie pojedynczych wag i biasów obrazujemy na modelu reprezentatywnym.
* **Trajektorie wybranych współczynników wagowych:**
  * Śledzenie wartości 4–6 pojedynczych wag $w_{ij}(t)$ z warstwy początkowej i końcowej w czasie uczenia, ilustrujące stabilizację parametrów na plateau.
* **Rozkłady wag (histogramy):**
  * Histogram wag przed rozpoczęciem uczenia (stan inicjalizacji) oraz po zakończeniu treningu dla warstw ukrytych i wyjściowej.

---

## 5. Część A: Eksperymenty Regresyjne ($\mathbb{R} \to \mathbb{R}$)

---

### Eksperyment R1: Wpływ funkcji aktywacji w warstwach ukrytych

#### 1. Cel i hipoteza
Zbadanie wpływu doboru nieliniowej funkcji aktywacji wewnątrz sieci na zdolność aproksymacji gładkich oraz wielomianowych funkcji jednoargumentowych.
* **Badane funkcje w warstwach ukrytych:** `Sigmoid`, `Tanh`, `ReLU`.
* **Warstwa wyjściowa:** Liniowa (brak aktywacji).
* **Hipoteza:**
  * `ReLU` będzie uczyć się szybko, lecz wygenerowana krzywa aproksymacji będzie odcinkowo liniowa (widoczne załamania/kanty w punktach zwrotnych funkcji `data.cube`).
  * `Tanh` i `Sigmoid`, jako funkcje gładkie ($C^\infty$), wygenerują naturalnie gładką krzywą dopasowania bez ostrych załamań, przy czym `Tanh` będzie zbiegać szybciej ze względu na symetrię wokół zera.

#### 2. Konfiguracja i zbiory danych
* **Zbiory danych:** `data.activation.train.1000.csv` / `test.1000.csv` oraz `data.cube.train.1000.csv` / `test.1000.csv`.
* **Architektura bazowa:** Stała, np. `[1, 32, 16, 1]`.
* **Parametry:** `lr = 0.01` (lub dostosowane do skali danych), `batch_size = 32`, `epochs = 150`, optymalizator `SGD`, `use_bias = True`.
* **Inicjalizacja wag:** `xavier` dla Sigmoid i Tanh, `he` dla ReLU.

#### 3. Wytwarzane wykresy (w języku polskim)
1. **Jakość aproksymacji funkcji:**
   * Wykres liniowy: *„Aproksymacja funkcji jednoargumentowej w zależności od funkcji aktywacji (Sigmoid vs ReLU vs Tanh)”* – krzywa modelu na tle punktów uczących i testowych.
2. **Krzywe zbieżności:**
   * Wykres liniowy: *„Przebieg funkcji straty (MSE) w kolejnych epokach dla badanych funkcji aktywacji”* (zbiór uczący i testowy).
3. **Ewolucja wag i gradientów:**
   * Wykres liniowy: *„Zmiana normy wag warstw ukrytych \|\|W\|\| w kolejnych epokach dla badanych funkcji aktywacji”*.
4. **Ewolucja wagi i biasu pierwszego neuronu:**
   * Wykres wielopanelowy 1x3: *„Ewolucja wagi i biasu pierwszego neuronu poszczególnych warstw (Sigmoid vs ReLU vs Tanh)”*.

---

### Eksperyment R2: Wpływ liczby warstw ukrytych oraz ich liczności

#### 1. Cel i hipoteza
Analiza pojemności aproksymacyjnej sieci perceptronowej w zależności od jej głębokości (od 0 do 4 warstw ukrytych) i szerokości w problemie regresji nieliniowej.
* **Hipoteza:**
  * **0 warstw ukrytych** (`[1, 1]`): Model zredukowany do klasycznej regresji liniowej. Nie będzie w stanie zamodelować zakrzywień wielomianowych ani sigmoidalnych (błąd zatrzyma się na wysokim poziomie).
  * **1 warstwa ukryta:** Zgodnie z twierdzeniem o uniwersalnej aproksymacji, wąska sieć (4 neurony) da zgrubne przybliżenie, natomiast szeroka (64 neurony) osiągnie wysoką dokładność, choć może wykazywać oscylacje na brzegach przedziału.
  * **2–4 warstwy ukryte:** Zwiększenie głębokości pozwoli na hierarchiczną reprezentację i dobre dopasowanie przy mniejszej łącznej liczbie wag, jednak sieć 4-warstwowa bez zaawansowanych optymalizatorów (Momentum/Adam) może uczyć się wolniej z powodu zanikania gradientów w pierwszych warstwach.

#### 2. Badane architektury
Należy przetestować warianty od 0 do 4 warstw ukrytych:
1. **0 warstw ukrytych:** `[1, 1]` (regresja liniowa)
2. **1 warstwa ukryta (wpływ szerokości):**
   * Wąska: `[1, 4, 1]`
   * Średnia: `[1, 16, 1]`
   * Szeroka: `[1, 64, 1]`
3. **2 warstwy ukryte:** `[1, 16, 8, 1]` oraz `[1, 32, 16, 1]`
4. **3 warstwy ukryte:** `[1, 32, 16, 8, 1]`
5. **4 warstwy ukryte:** `[1, 64, 32, 16, 8, 1]`

* **Parametry stałe:** Najlepsza funkcja aktywacji z R1 (np. `Tanh` lub `ReLU`), funkcja straty `MSELoss`, `batch_size = 32`, `epochs = 150`, stały `learning_rate`.

#### 3. Wytwarzane wykresy (w języku polskim)
1. **Porównanie krzywych aproksymacji:**
   * Wykres wielopanelowy (grid subplotów): *„Wpływ głębokości architektury na jakość aproksymacji funkcji (od 0 do 4 warstw ukrytych)”*.
2. **Błąd końcowy w zależności od złożoności modelu:**
   * Wykres słupkowy / punktowy: *„Wartość błędu MSE na zbiorze testowym w zależności od liczby warstw ukrytych”*.
3. **Dynamika uczenia:**
   * Wykres liniowy: *„Krzywe uczenia (błąd w epokach) dla architektur o różnej głębokości”*.
4. **Wizualizacja wag dla sieci najgłębszej (4 warstwy):**
   * Wykres liniowy: *„Ewolucja normy wag poszczególnych warstw w sieci 4-warstwowej”* (porównanie pierwszej warstwy ukrytej z ostatnią).
5. **Ewolucja wagi i biasu pierwszego neuronu:**
   * Wykres wielopanelowy 1x3: *„Ewolucja wagi i biasu pierwszego neuronu poszczególnych warstw w zależności od architektury”*.

---

### Eksperyment R3: Wpływ miary błędu na wyjściu sieci (MSE vs MAE)

#### 1. Cel i hipoteza
Porównanie dynamiki uczenia perceptronu przy optymalizacji błędu średniokwadratowego (`MSELoss` / $L_2$) oraz średniego błędu bezwzględnego (`MAELoss` / $L_1$).
* **Hipoteza:**
  * Gradient błędu MSE jest proporcjonalny do wielkości rezyduum ($2(y - \hat{y})$), co skutkuje silnymi korektami wag przy dużych błędach, ale spowolnieniem zbieżności w bezpośrednim sąsiedztwie minimum.
  * Gradient MAE ma stały moduł ($\pm 1$), co zapewnia stałe tempo ruchu wag niezależnie od odległości od celu, jednak wokół minimum może powodować oscylacje (chyba że współczynnik uczenia maleje). Ponadto MAE wykazuje mniejszą wrażliwość na ewentualne punkty odstające (*outliers*).

#### 2. Konfiguracja
* **Porównywane funkcje celu:** `MSELoss` vs `MAELoss`.
* **Dane:** `data.cube` oraz `data.activation` ($N=1000$).
* **Architektura:** Stała (np. `[1, 32, 16, 1]`, aktywacja `Tanh` lub `ReLU`).
* **Testowane tempo uczenia:** Sprawdzenie dla 2–3 wartości `learning_rate`, gdyż stały gradient MAE często wymaga innego kroku niż gradient MSE zależny od skali.

#### 3. Wytwarzane wykresy (w języku polskim)
1. **Dopasowanie do danych:**
   * Wykres liniowy: *„Porównanie funkcji aproksymacyjnej uzyskanej z minimalizacji MSE oraz MAE”*.
2. **Dynamika błędu w skali iteracji i epok:**
   * Wykres liniowy: *„Przebieg straty w kolejnych iteracjach: MSE vs MAE”* (znormalizowany lub przedstawiony w niezależnych skalach).
3. **Stabilność parametrów:**
   * Wykres liniowy: *„Trajektoria wybranej wagi w czasie: oscylacje MAE vs zbieżność MSE”*.
4. **Ewolucja wagi i biasu pierwszego neuronu:**
   * Wykres wielopanelowy 1x3: *„Ewolucja wagi i biasu pierwszego neuronu poszczególnych warstw (MSE vs MAE)”*.

---

### Eksperyment R4 (Dodatkowy): Wpływ rozmiaru zbioru danych uczących

#### 1. Cel i hipoteza
Zbadanie wpływu liczności zbioru treningowego $N \in \{100, 500, 1000, 10000\}$ na zdolność generalizacji modelu regresyjnego i zjawisko luki generalizacyjnej (*generalization gap*).
* **Hipoteza:** Dla $N=100$ model o dużej liczbie parametrów ulegnie częściowemu przeuczeniu (dopasowanie do szumu, wysoki błąd testowy). Wraz ze wzrostem $N$ luka pomiędzy błędem uczącym a testowym drastycznie maleje, stabilizując estymowaną krzywą.

#### 2. Wytwarzane wykresy (w języku polskim)
* Wykres liniowy / punktowy: *„Błąd MSE na zbiorze uczącym i testowym w funkcji wielkości próby N (Krzywa uczenia modelu)”*.
* Zestawienie 4 podwykresów: *„Kształt dopasowanej krzywej regresji dla próby N = 100, 500, 1000, 10000”*.

---

## 6. Część B: Eksperymenty Klasyfikacyjne ($\mathbb{R}^2 \to \mathcal{C}$)

---

### Eksperyment K1: Wpływ funkcji aktywacji w warstwach ukrytych

#### 1. Cel i hipoteza
Ocena wpływu funkcji aktywacji w warstwach ukrytych (`Sigmoid`, `Tanh`, `ReLU`) na geometrię granic decyzyjnych oraz prędkość zbieżności w zadaniu separacji klas w przestrzeni 2D.
* **Warstwa wyjściowa:** `Sigmoid` (dla klasyfikacji binarnej `data.simple`) lub `Softmax` (dla wieloklasowej `data.three_gauss`).
* **Hipoteza:** `ReLU` zapewni najszybszą redukcję straty i odporność na zanikający gradient, tworząc łamane (odcinkowo liniowe) granice decyzyjne. `Sigmoid` w warstwach ukrytych będzie zbiegać najwolniej. `Tanh` wygeneruje gładkie granice decyzyjne i osiągnie dobrą dynamikę dzięki centrowaniu wokół zera.

#### 2. Konfiguracja
* **Zbiory danych:** `data.simple.train.100.csv` / `test.100.csv` oraz `data.three_gauss.train.100.csv` / `test.100.csv`.
* **Architektura:** Stała, np. `[2, 32, 16, C]`, gdzie $C \in \{1, 3\}$ w zależności od zadania.
* **Parametry:** `lr = 0.05`, `batch_size = 32`, `epochs = 150`, optymalizator `SGD`, `use_bias = True`.
* **Inicjalizacja:** `xavier` dla Sigmoid i Tanh, `he` dla ReLU.

#### 3. Wytwarzane wykresy (w języku polskim)
1. **Granice decyzyjne w 2D:**
   * Zestawienie podwykresów: *„Granice decyzyjne w przestrzeni cech dla funkcji aktywacji (Sigmoid vs ReLU vs Tanh)”* wraz z nałożonymi punktami testowymi.
2. **Krzywe uczenia i metryki:**
   * Wykres liniowy: *„Funkcja straty w kolejnych epokach dla badanych funkcji aktywacji”*.
   * Wykres liniowy: *„Dokładność klasyfikacji (Accuracy [%]) na zbiorze testowym w funkcji epok”*.
3. **Normy wag:**
   * Wykres liniowy: *„Ewolucja normy wag \|\|W\|\| w kolejnych epokach dla badanych funkcji aktywacji”*.
4. **Ewolucja wagi i biasu pierwszego neuronu:**
   * Wykres wielopanelowy 1x3: *„Ewolucja wagi i biasu pierwszego neuronu poszczególnych warstw (Sigmoid vs ReLU vs Tanh)”* (dla reprezentatywnego ziarna).

---

### Eksperyment K2: Wpływ liczby warstw ukrytych oraz ich liczności

#### 1. Cel i hipoteza
Analiza zdolności dyskryminacyjnej perceptronu przy przejściu od modelu liniowego (0 warstw ukrytych) do sieci głębokich (do 4 warstw ukrytych).
* **Hipoteza:**
  * 0 warstw ukrytych (`[2, C]`): Wyznacza wyłącznie prostoliniowe hiperpłaszczyzny rozdzielające; nie poradzi sobie z nieliniowymi granicami w `three_gauss` ani wygięciami w `simple`.
  * 1–2 warstwy ukryte: Pozwolą na pełne i płynne odseparowanie klas.
  * 3–4 warstwy ukryte: Wykażą zdolność tworzenia bardzo złożonych granic, ale przy prostym SGD mogą wykazywać wolniejszy start zbieżności w początkowych warstwach.

#### 2. Badane architektury
1. **0 warstw ukrytych:** `[2, C]` (model liniowy / regresja logistyczna)
2. **1 warstwa ukryta:**
   * Wąska: `[2, 4, C]`
   * Średnia: `[2, 16, C]`
   * Szeroka: `[2, 64, C]`
3. **2 warstwy ukryte:** `[2, 16, 8, C]` oraz `[2, 32, 16, C]`
4. **3 warstwy ukryte:** `[2, 32, 16, 8, C]`
5. **4 warstwy ukryte:** `[2, 64, 32, 16, 8, C]`

* **Parametry stałe:** Wybrana najlepsza aktywacja z K1, `CrossEntropyLoss`, `batch_size = 32`, `epochs = 150`, stały `learning_rate`.

#### 3. Wytwarzane wykresy (w języku polskim)
1. **Wpływ architektury na granice decyzyjne:**
   * Siatka wykresów (grid): *„Ewolucja kształtu granic decyzyjnych w zależności od liczby warstw ukrytych (0–4 warstw)”*.
2. **Skuteczność klasyfikacji:**
   * Wykres słupkowy: *„Dokładność klasyfikacji na zbiorze testowym w zależności od architektury sieci”*.
3. **Zbieżność w epokach:**
   * Wykres liniowy: *„Przebieg funkcji straty dla architektur o głębokości od 0 do 4 warstw”*.
4. **Ewolucja wag w sieci głębokiej:**
   * Wykres liniowy: *„Ewolucja norm wag poszczególnych warstw w sieci 4-warstwowej”*.
5. **Ewolucja wagi i biasu pierwszego neuronu:**
   * Wykres wielopanelowy 1x3: *„Ewolucja wagi i biasu pierwszego neuronu poszczególnych warstw w zależności od architektury”*.

---

### Eksperyment K3: Wpływ miary błędu na wyjściu sieci (Cross-Entropy vs MSE)

#### 1. Cel i hipoteza
Porównanie skuteczności optymalizacji funkcji entropii krzyżowej (`CrossEntropyLoss`) oraz błędu średniokwadratowego (`MSELoss`) w zadaniu klasyfikacji z nieliniowymi funkcjami wyjściowymi (`Sigmoid` / `Softmax`).
* **Hipoteza:** Entropia krzyżowa jest optymalna pod kątem probabilistycznym. W połączeniu z sigmoidalnym/softmaxowym wyjściem, funkcja MSE wywołuje zjawisko *nasycenia gradientu* (gradient saturation): przy błędnej predykcji o wysokiej pewności pochodna aktywacji dąży do zera, co paraliżuje proces aktualizacji wag. Entropia krzyżowa kompensuje ten efekt, gwarantując duży gradient i szybszą zbieżność.

#### 2. Konfiguracja
* **Porównywane funkcje celu:**
  * Dla `data.simple`: `BinaryCrossEntropyLoss` vs `MSELoss`.
  * Dla `data.three_gauss`: `CategoricalCrossEntropyLoss` vs `MSELoss` (cele w one-hot).
* **Architektura:** Stała, np. `[2, 32, 16, C]`, identyczne wagi początkowe (`set_seed`).
* **Parametry:** `batch_size = 32`, `epochs = 150`, porównanie dla kilku wartości `learning_rate`.

#### 3. Wytwarzane wykresy (w języku polskim)
1. **Krzywe uczenia:**
   * Wykres liniowy: *„Porównanie zbieżności funkcji straty i dokładności: Entropia Krzyżowa vs Błąd Średniokwadratowy (MSE)”*.
2. **Kształt granic decyzyjnych:**
   * Zestawienie 2D: *„Granice decyzyjne wyznaczone przy optymalizacji Entropią Krzyżową oraz MSE”*.
3. **Dynamika gradientów:**
   * Wykres liniowy: *„Norma aktualizacji wag w początkowych epokach: Cross-Entropy vs MSE”* (uwidocznienie ucieczki ze stanu nasycenia).
4. **Ewolucja wagi i biasu pierwszego neuronu:**
   * Wykres wielopanelowy 1x3: *„Ewolucja wagi i biasu pierwszego neuronu poszczególnych warstw (Cross-Entropy vs MSE)”*.

---

## 7. Zbiorcza macierz konfiguracji eksperymentów

Poniższa tabela stanowi jednoznaczne podsumowanie dla procedur implementacyjnych:

| Identyfikator | Zadanie | Cel badania | Zmienna badana | Wartości zmiennej | Wyjście sieci / Funkcja celu | Metryki i wizualizacje |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | Regresja | Funkcje aktywacji | Aktywacja ukryta | `Sigmoid`, `Tanh`, `ReLU` | Linear / `MSELoss` | MSE, aproksymacja, $\|W\|$, wagi neuronu 0 |
| **R2** | Regresja | Architektura sieci | Głębokość i szerokość | 0–4 warstwy ukryte (od `[1,1]` do `[1,64,32,16,8,1]`) | Linear / `MSELoss` | MSE, $R^2$, krzywa 1D, $\|W\|$, wagi neuronu 0 |
| **R3** | Regresja | Miara błędu | Funkcja straty | `MSELoss` vs `MAELoss` | Linear | MSE, MAE, odporność, wagi neuronu 0 |
| **R4** | Regresja | Rozmiar próby | Liczność $N$ | 100, 500, 1000, 10000 | Linear / `MSELoss` | Błąd generalizacji, MSE, krzywa 1D |
| **K1** | Klasyfikacja | Funkcje aktywacji | Aktywacja ukryta | `Sigmoid`, `Tanh`, `ReLU` | Sigmoid/Softmax / `CrossEntropy` | Accuracy [%], Loss, Granice 2D, $\|W\|$, wagi neuronu 0 |
| **K2** | Klasyfikacja | Architektura sieci | Głębokość i szerokość | 0–4 warstwy ukryte (od `[2,C]` do `[2,64,32,16,8,C]`) | Sigmoid/Softmax / `CrossEntropy` | Accuracy [%], Granice 2D, $\|W\|$, wagi neuronu 0 |
| **K3** | Klasyfikacja | Miara błędu | Funkcja straty | `CrossEntropyLoss` vs `MSELoss` | Sigmoid/Softmax | Accuracy [%], tempo zbieżności, $\|W\|$, wagi neuronu 0 |

---

## 8. Wytyczne organizacyjne dla kodu eksperymentów

W celu zachowania przejrzystości repozytorium zaleca się następującą strukturę katalogów dla modułu eksperymentów:

```
experiments/
├── common/
│   ├── plotting.py          # Zunifikowane funkcje wizualizacji (Trzy filary, etykiety PL)
│   └── data_loader.py       # Standaryzacja i ładowanie zbiorów z data/p1/
├── run_regression.py        # Skrypt uruchamiający eksperymenty R1 - R4
├── run_classification.py    # Skrypt uruchamiający eksperymenty K1 - K4
└── results/
    ├── regression/          # Zapisane wykresy .png i metryki .csv dla regresji
    └── classification/      # Zapisane wykresy .png i metryki .csv dla klasyfikacji
```

---
