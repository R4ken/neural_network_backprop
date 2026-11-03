# Odtworzenie wyników MNIST

Z katalogu głównego repozytorium:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python experiments/run_mnist.py
```

Domyślnie skrypt trenuje 10 modeli przez 30 epok dla ziaren `54217`–`54226`.
Podział danych jest stały (ziarno `54217`): 55 000 próbek uczących,
5 000 walidacyjnych i pełny, niezależny test liczący 10 000 próbek.
Liczebności są wypisywane po wczytaniu danych i zapisywane w podsumowaniu.

W `experiments/results/mnist/` powstają:

- `mnist_learning_curves_mean_std.png`: średnia i pasma ±1 odchylenia
  standardowego między ziarnami, osobno dla każdej epoki i metryki;
- `mnist_training_summary.json`: konfiguracja, rzeczywiste liczebności,
  historie każdego ziarna, wyniki testowe, średnie i odchylenia (`ddof=0`);
- macierz pomyłek, tabela metryk, przykłady błędów i pojedyncze krzywe
  dla pierwszego ziarna (`54217`). Nie są to wyniki uśrednione.

Podsumowanie zapisuje postęp po każdym ukończonym ziarnie; pola `mean` i `std`
oraz wykres zbiorczy powstają dopiero po ukończeniu wszystkich uruchomień.
Katalog wyników jest ignorowany przez Git, więc na nowej kopii repozytorium
należy najpierw uruchomić eksperyment.

Kompilacja raportu po zakończeniu obliczeń, z katalogu `mnist_report/`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Test techniczny: dodaj `--quick`. Używa 1 000 próbek uczących, 200 walidacyjnych,
500 testowych i jednej epoki; domyślnie zapisuje do osobnego katalogu
`experiments/results/mnist/quick/`, nie nadpisując wyników raportu.
Opcje `--seeds`, `--split-seed` i `--results-dir` pozwalają zmienić konfigurację;
wyników z inną konfiguracją nie należy przedstawiać jako domyślnego eksperymentu.
