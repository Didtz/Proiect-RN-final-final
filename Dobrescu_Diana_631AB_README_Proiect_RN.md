# Plant Identifier - Aplicatie Web pentru Identificarea Plantelor

## Ce este aceasta aplicatie?

Aceasta este o aplicatie web completa pentru identificarea plantelor folosind inteligenta artificiala. Puteti incarca o fotografie cu o planta si aplicatia va identifica specia plantei cu un procent de incredere.

## Caracteristici principale

- Identificarea plantelor din fotografie folosind retele neuronale profunde
- Afisarea unei imagini de referinta pentru comparatie
- Informatii detaliate despre planta (nume stiintific, ingrijire, cerinte de lumina si apa)
- Interfata web moderna si responsiva
- Design minimalist in alb si negru

## Instalare

### Pasul 1: Instalarea dependentelor Python

```bash
pip install -r requirements.txt
```

Aceasta va instala: Flask, TensorFlow, Keras, NumPy, Pillow si alte pachete necesare.

### Pasul 2: Antrenarea modelului (prima data)

```bash
python train_model.py
```

Acest pas va antrena modelul de invatare automata pe plantele din dosarul training_data/. Dureza aproximativ 10-15 minute.

### Pasul 3: Pornirea aplicatiei

```bash
python app_improved.py
```

Dupa aceasta, deschideti in browser: http://localhost:5000

## Cum se foloseste

1. Deschideti aplicatia in browser la adresa http://localhost:5000
2. Navigati la sectiunea "Identify"
3. Incarcati o fotografie cu o planta (PNG, JPG, JPEG sau GIF)
4. Faceti clic pe butonul "Identify Plant"
5. Aplicatia va afisa:
   - Fotografia incarcata
   - O imagine de referinta din setul de antrenare
   - Numele plantei si numele stiintific
   - Informatii despre ingrijire (apa, lumina, nivel dificultate)
   - Procentul de incredere al identificarii

## Structura fisierelor

```
.
├── app_improved.py              Aplicatia Flask principala
├── train_model.py               Script de antrenare a modelului
├── requirements.txt             Dependentele Python
├── templates/                   Fisiere HTML
│   ├── index.html              Pagina de start
│   ├── identify.html           Pagina de identificare
│   ├── database.html           Baza de date cu plante
│   └── guide.html              Ghid de ingrijire
├── static/
│   └── style.css               Stiluri CSS
├── models/                      Modele antrenate
│   └── plant_model.h5          Modelul antrenat
├── training_data/              Imagini de referinta pentru plante
│   ├── cactus/
│   ├── orchid/
│   ├── rose/
│   ├── sunflower/
│   └── tulip/
└── uploads/                    Folder temporar pentru imagini incarcate
```

## API Endpoints

### POST /api/predict

Incarca o imagine si returneaza predictia plantei.

Request:
```
FormData cu campul "file" continand fisierul imagine
```

Response:
```json
{
  "success": true,
  "plant": "rose",
  "common_name": "Rose",
  "confidence": 0.85,
  "info": {
    "scientific_name": "Rosa spp.",
    "description": "...",
    "watering": "...",
    "light": "...",
    "difficulty": "..."
  },
  "comparison_image": "data:image/jpeg;base64,..."
}
```

### GET /api/plants

Returneaza lista cu toate plantele disponibile.

### GET /api/plant/<name>

Returneaza informatii despre o planta specifica.

### GET /api/model/status

Verifica daca modelul a fost antrenat.

## Configurare

### Schimbarea portului

Deschideti fisierul `app_improved.py` si modificati linia finala:

```python
app.run(debug=True, port=5000, host='0.0.0.0')
```

Inlocuiti 5000 cu portul dorit.

### Adaugarea de noi plante

1. Creati un folder cu numele plantei in dosarul `training_data/`
2. Adaugati imagini ale plantei in acel folder
3. Rulati: `python train_model.py`

Aplicatia va detecta automat noua planta.

## Rezolvarea problemelor

### Eroare: "Model not found"

Solutie: Rulati mai intai `python train_model.py`

### Eroare: "Port 5000 already in use"

Solutie: Schimbati portul in fisierul `app_improved.py` sau inchideti aplicatia care foloseste portul 5000.

### Eroare: "ModuleNotFoundError"

Solutie: Reinstalati dependentele cu `pip install -r requirements.txt`

### Predictii lente

Normal este ca prima predictie sa dureze 3-5 secunde (modelul se incarca). Predictiile urmatoare sunt mai rapide (1-2 secunde).

## Cerinte sistem

- Python 3.8 sau mai nou
- 4 GB RAM minim (8 GB recomandat)
- 2 GB spatiu liber pe disc
- Browser modern (Chrome, Firefox, Safari, Edge)

## Tehnologii folosite

- Backend: Flask 2.3.3
- Invatare automata: TensorFlow 2.13.0, Keras 2.13.1
- Arhitectura model: MobileNetV2 cu transfer learning
- Frontend: HTML5, CSS3, JavaScript
- Baze de date imagine: NumPy, Pillow

## Performanta

- Dimensiune model: aproximativ 85-100 MB
- Timp de predictie: sub 1 secunda (dupa incarcarea modelului)
- Timp de incarcarea model: 2-3 secunde (prima data)
- Consum RAM: 500 MB - 2 GB
- Support GPU: Automat (daca CUDA este disponibil)

## Detalii model antrenare

Modelul foloseste transfer learning cu MobileNetV2:

1. Baza: MobileNetV2 preantrenat pe ImageNet
2. Straturi personalizate:
   - Global Average Pooling
   - Dense 256 cu ReLU si Dropout 0.5
   - Dense 128 cu ReLU si Dropout 0.3
   - Dense N cu Softmax (N = numar clase)

3. Data Augmentation:
   - Rotatie: 40 grade
   - Zoom: 0.2
   - Shift: 0.2
   - Flip orizontal si vertical

4. Callbacks:
   - Early Stopping (patience 3)
   - Learning Rate Reduction
   - Model Checkpointing

## Extensii posibile

1. Adaugati mai multe plante in dosarul training_data/
2. Antrenati modelul cu mai multe imagini pentru mai buna acuratete
3. Adaugati autentificare utilizatori
4. Stocati istoricul predictiilor in baza de date
5. Creeati aplicatie mobila
6. Implementati webcam live identification
7. Deplorati pe cloud (AWS, Heroku, Azure)

## Utilizare pentru primii utilizatori

1. Instalati dependentele: `pip install -r requirements.txt`
2. Antrenati modelul: `python train_model.py` (asteptati 10-15 minute)
3. Porniti aplicatia: `python app_improved.py`
4. Deschideti in browser: http://localhost:5000
5. Incarcati o fotografie cu o planta si vedeti rezultatul

## Support

Pentru probleme tehnice:

1. Verificati ca Python 3.8+ este instalat
2. Verificati ca toate dependentele sunt instalate cu `pip list`
3. Asigurati-va ca dosarul training_data/ contine imagini
4. Consultati log-urile din terminal pentru erori specifice

## Licenta

Aceasta aplicatie a fost creata pentru scopuri educationale.

## Autori

Proiect de inteligenta artificiala pentru identificarea plantelor
Data: februarie 2026
