'''
MuseumLangAPI: REST API per il riconoscimento automatico della lingua di testi museali.

Ho integrato spiegazioni dettagliate per ogni scelta architetturale e logica.
''' 
import logging
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Configuro il logging su file per motivi di audit e debug.
#    Ho scelto il livello INFO per tracciare tutte le richieste/risposte,
#    includendo timestamp e livello di severità.
logging.basicConfig(
    filename="museumlang_api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 2. Definisco i modelli Pydantic per validazione automatica e documentazione.
class LanguageRequest(BaseModel):
    # Il campo 'text' contiene il testo da analizzare.
    # Uso Pydantic per assicurarmi che arrivi sempre una stringa.
    text: str

class LanguageResponse(BaseModel):
    # Il codice ISO della lingua individuata e il punteggio di confidenza.
    language_code: str
    confidence: float

# 3. Inizializzo FastAPI.
#    Scelgo FastAPI perché fornisce gestione asincrona,
#    validazione tramite Pydantic e auto-generazione di docs OpenAPI.
app = FastAPI(
    title="MuseumLangAPI",
    description="API per il riconoscimento automatico della lingua di testi museali, completa di commenti esplicativi.",
    version="1.0.0"
)

# 4. Carico il modello all'avvio, in un blocco try-except.
#    In questo modo garantisco che, se il modello non è disponibile,
#    l'applicazione non parta in uno stato inconsistente.
try:
    with open("language_detection_pipeline.pkl", "rb") as f:
        loaded_pipeline = pickle.load(f)
    logging.info("[Init] Modello di rilevamento lingua caricato con successo.")
except Exception as e:
    # Se fallisce il caricamento, loggo l'errore ed esco immediatamente.
    logging.error(f"[Init] Errore caricamento modello: {e}")
    raise RuntimeError(f"Impossibile caricare il modello: {e}")

@app.post("/identify-language", response_model=LanguageResponse)
async def identify_language(request: LanguageRequest):
    '''
    Endpoint per identificare la lingua di un testo.

    Logica interna e motivazioni:
    1. Sanitizzo il testo con strip() per rimuovere spazi superflui.
    2. Se il testo è vuoto, rispondo con 400 per evitare chiamate inutili.
    3. Se il pipeline supporta predict_proba (probabilità), lo uso per ottenere confidenza.
       Questo mi consente di restituire un valore di confidenza reale, non solo 1.0.
    4. In caso contrario, uso predict() e imposto confidenza a 1.0 (fallback).
    5. Registro ogni richiesta e risposta per audit e analisi futura.
    6. Gestisco errori generici restituendo 500.
    '''
    # 1. Pulizia input
    text = request.text.strip()
    if not text:
        # 2. Testo vuoto -> errore 400
        logging.warning(f"[BadRequest] Testo vuoto ricevuto: {request.text!r}")
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto.")

    try:
        # 3. Uso predict_proba se disponibile per calcolare la confidenza.
        if hasattr(loaded_pipeline, "predict_proba"):
            probabilities = loaded_pipeline.predict_proba([text])[0]
            classes = loaded_pipeline.classes_
            idx = int(np.argmax(probabilities))
            language_code = classes[idx]
            confidence = float(probabilities[idx])
        else:
            # 4. Fallback a predict senza confidenza
            language_code = loaded_pipeline.predict([text])[0]
            confidence = 1.0

        # Costruisco risposta
        response = LanguageResponse(language_code=language_code, confidence=confidence)
        # 5. Logging della richiesta e risposta
        logging.info(f"[Predict] '{text}' -> {response.json()}")
        return response

    except Exception as e:
        # 6. Errore interno -> 500
        logging.error(f"[Error] Durante la predizione: {e}")
        raise HTTPException(status_code=500, detail="Errore interno nel riconoscimento della lingua.")

@app.get("/", tags=["Health Check"])
async def root():
    '''
    Health check endpoint.
    Motivo: utile per monitoring e verificare rapidamente se l'API è attiva.
    '''
    return {"message": "MuseumLangAPI è attiva."}

# 7. Punto di ingresso per eseguire il server con uvicorn.
#    Uso reload=True in sviluppo per ricaricare automaticamente il codice.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("museum_lang_api:app", host="0.0.0.0", port=8000, reload=True)
