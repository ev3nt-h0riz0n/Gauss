# Projekt CUDA - szum gaussowski

## Wstęp
Celem projektu było zaprojektowanie oraz zaimplementowanie filtru Gaussa, z wykorzystaniem architektury GPU firmy NVIDIA. Implementacja została wykonana w języku CUDA C++. Szybkość i jakość działania programu została porównana do odpowiednika z biblioteki OpenCV.

---

## Opis algorytmu
Rozmycie Gaussa (inaczej nazywane filtrem Gaussa), to operacja modyfikacji obrazu poprzez jego wizualne rozmycie. Stosuje się je w celu zmniejszenia zakłóceń występujących na obrazie oraz wygładzenia jego ostrych krawędzi i przejść kolorów.

Sam algorytm polega na analizie każdego pojedynczego piksela obrazu, biorąc pod uwagę jego sąsiadów. Dla każdego piksela obliczana jest średnia ważona sąsiednich wartości. Wagi te są określane przez kernel Gaussa o zadanych przez użytkownika wymiarach.

W celu przyspieszenia programu, postanowiono użyć separowalności filtru Gaussa. Oznacza to, że zamiast wielokrotnego użycia dwuwymiarowego kernela, używa się operacji rozmycia w pionie oraz poziomie. Wydajność w takim przypadku znacząco wzrasta, ponieważ algorytm 2D generuje $N \times N$ operacji, natomiast splot dwóch operacji 1D generuje $2 \times N$ operacji, gdzie $N$ oznacza wymiar kernela. Warto zaznaczyć, że zastosowanie separowalności w algorytmie nie zmienia jakości wynikowej filtru. Przykładowe porównanie metod zostało przedstawione na poniższym rysunku na przykładzie kernela 5:5 pikseli.

(wkleić obraz)

### Wzory
Filtracja pozioma oraz pionowa opisane są wzorami:
$$G(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{x^{2}}{2\sigma^{2}}}$$
$$G(y)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{y^{2}}{2\sigma^{2}}}$$
gdzie:
*   $(x,y)$ - współrzędne piksela
*   $\sigma$ - odchylenie standardowe

W algorytmie wykonywana jest operacja konwolucji obrazu z wygenerowanym kernelem, polegająca na przesunięcie go po całym obszarze obrazu oraz obliczeniu nowych wartości każdego piksela, jako sumę iloczynów wartości sąsiednich pikseli i ich wag.

Pozioma konwolucja 1D opisana jest wzorem:
$$I_{H}=\sum_{i=-k}^{k}I_{0}(x+i,y) \cdot G(i)$$
$$I_{V}=\sum_{j=-k}^{k}I_{H}(x,y+j) \cdot G(j)$$
gdzie:
*   $(x,y)$ - współrzędne piksela
*   $I_{0}$ - początkowa wartość piksela
*   $I_{H}$ - wartość piksela po filtracji poziomej
*   $I_{V}$ - wartość piksela po filtracji poziomej i pionowej
*   $G(i)$ - wartość poziomego kernela Gaussa
*   $G(j)$ - wartość pionowego kernela Gaussa

### Wywołanie
Aby wywołać funkcję należy użyć:
`GaussianBlurFastCV(src, des, cv::Size(M,N), sigma);`

gdzie:
*   `src` - zdjęcie źródłowe
*   `des` - nazwa zdjęcia wynikowego
*   `cv::Size(M,N)` - rozmiar maski Gaussa, gdzie M, N to jego wymiary
*   `sigma` - moc z jaką algorytm ma działać na obraz

Przy użyciu niesymetrycznej maski (M=/=N) należy spodziewać się nierównomiernego rozmycia obrazu.
---

## 3. Uzasadnienie wyboru odpowiednich technik

*   **Thrust:**
    *   `Gauss_gen` - użyty do obliczenia wartości gaussa dla wektora o zadanej wielkości.
    *   `Gauss_gen_norm` - użyty do uzupełnienia kernela przeskalowanymi wartościami filtru Gaussa.
    *   `counting_iterator` - ominięcie zapisania w pamięci liczb od 0-size.
    *   `make_transform_iterator` - generacja wektora z wartościami gaussa bez zapisywania go w pamięci.
    *   `reduce` - zsumowanie wartości z wektora.
    *   `transform` - generuje wektor z znormalizowanymi wartościami do używania w filtrze.
    *   `raw_pointer_cast` - przerzucenie danych z transforma do kernela.
    *   `pair` - by złączone elementy przenosić tylko jednokrotnie, zamiast dwukrotnie.
    *   Celem użycia biblioteki thrust jest ograniczenie zapisywania danych używanych podczas obliczeń do pamięci.
*   **Asynchroniczność:** Dodanie asynchronicznego kopiowania pamięci ma na celu umożliwienie CPU wykonywania innych operacji podczas przesyłu danych.
*   **Shared memory:** Zastosowano w celu uniknięcia wielokrotnego sięgania do pamięci globalnej. Zrealizowano, to poprzez załadowanie danej części zdjęcia - tile - jednokrotnie przez wszystkie wątki zawarte w bloku.

---

## 4. Analiza wydajności

Podczas wykonywania pomiarów zauważono, że sama inicjalizacja biblioteki CUDA oraz kerneli zajmuje bardzo dużo czasu (20ms-40ms). W związku z tym przed porównaniem z OpenCV jednokrotnie wywołano funkcję FastCV, aby pomiary były miarodajne. Wszystkie pomiary były realizowane na tym samym zdjęciu o wymiarach $608 \times 608$ oraz kernelu o wymiarach $33 \times 33$. Pomiary były wykonywane poprzez platformę Collab używając GPU 4T.

### 4.1. wersja1.cu
Wersja pierwsza FastCV nie zawiera żadnych elementów CUDA. Ponadto w algorytmie maska Gauss'a jest dwuwymiarowa.
*   Czas wykonania OpenCV: 0.0214444s
*   Czas wykonania FastCV: 12.291s

### 4.2. wersja2.cu
Zastosowano separowalność kernela Gaussa oraz użyto kerneli CUDA.
*   Czas wykonania OpenCV: 0.0101817s
*   Czas wykonania FastCV: 0.00188767s

### 4.3. wersja3.cu
Użyto biblioteki thrust, pozwalającej na implementację mechanizmów automatycznie zarządzających pamięcią.
*   Czas wykonania OpenCV (bez profilowania): 0.0100057s
*   Czas wykonania FastCV (bez profilowania): 0.0020681s

### 4.4. wersja4.cu
Zastosowano programowanie asynchroniczne.
*   Czas wykonania OpenCV: 0.0152528s
*   Czas wykonania FastCV: 0.00250476s

### 4.5. wersja5.cu - wersja ostateczna
Zastosowano shared memory w kernelu wertykalnym oraz horyzontalnym.
*   Czas wykonania OpenCV: 0.0101313s
*   Czas wykonania FastCV: 0.00218799s

---

## 5. Wnioski projektowe

*   Inicjalizacja CUDA oraz kerneli zajmuje bardzo dużo czasu, w związku z czym na początku każdego programu należałoby zainicjalizować rozgrzewkowe wykonanie funkcji.
*    Metoda FastCV pokazuje przewagę dopiero przy obrazach lub kernelach o większym rozmiarze, gdzie równoległe działanie wątków może zdominować działanie na procesorze.
*  Profilowanie przez Nsight Systems spowalnia pracę programu, ponieważ podczas pomiaru czasowego chrono zapisuje zarejestrowane przez siebie pomiary.
*   Thrust pozwala na automatyczne zarządzanie pamięcią i generację kernela bezpośrednio na GPU, jednak może powodować wielokrotne wywoływanie kerneli oraz częstą alokację pamięci.
*   Największy udziałem czasowym nie są obliczenia, a operacje pomocnicze, takie jak: alokacja, kopiowanie danych D2H/H2D oraz synchronizacja. Z tego powodu należałoby w przyszłości zmniejszyć ilość takich operacji, aby uzyskać lepszą wydajność.
*   Shared memory optymalizuje dostęp wątków do informacji zawartych w danym bloku, co powinno przyspieszyć wykonywane obliczenia.
*   Wersje 2-5 cechują się podobną wydajnością, ponieważ mimo zmian,problem leży w zarządzaniu pamięcią. By usprawnić działanie kodu, należałoby poświęcić więcej uwagi kwestii ograniczenia alokacji oraz transferów między-pamięciowych. Dobrym rozwiązaniem tego problemu mogło być użycie biblioteki CUB. Thrust jest biblioteką wysokopoziomową, utworzoną na bibliotece CUB, która zapewnia lepszą kontrolę nad ręcznym zarządzaniem pamięcią. Zatem unikniętoby niejawnych synchronizacji oraz alokacji.

---

## 6. Główne źródła
*   https://www.crisluengo.net/archives/22/
*   https://en.wikipedia.org/wiki/Gaussian_blur
*   https://www.cs.put.poznan.pl/kkrawiec/wiki/uploads/Zajecia/po3.pdf
*   https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
*   https://www.youtube.com/watch?v=cOBtkPsgkus
*   https://www.youtube.com/watch?v=pyW9St8uM8w
*   https://docs.nvidia.com/nsight-systems/UserGuide/index.html
*   https://forums.developer.nvidia.com/t/device-initialization-takes-60-seconds/257795/4
*   https://modal.com/gpu-glossary/
