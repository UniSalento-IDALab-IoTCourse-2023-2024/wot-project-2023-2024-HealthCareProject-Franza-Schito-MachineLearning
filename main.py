from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import json
from pydantic import BaseModel
from model_randomForest import load_and_predict

app = FastAPI()

class ActivityData(BaseModel):
    step: int
    caloriesOut: int

@app.post("/model")
async def model(data: ActivityData):
    try:
        print(f"JSON ricevuto: {data}")

        steps = data.step
        calories_out = data.caloriesOut

        if steps is None or calories_out is None:
            raise HTTPException(status_code=400, detail="Missing required fields in JSON")

        output = pd.DataFrame({
            'Steps': [steps],
            'Calories_Out': [calories_out]
        })

        output.to_csv('output.csv', index=False)
        print("CSV creato con successo")

        df = pd.read_csv('output.csv')
        print(f"DataFrame letto: {df}")

        result = load_and_predict("random_forest_model.pkl", df)
        print(f"Risultato modello: {result}")

        try:
            if isinstance(result, list) and len(result) > 0:
                prediction_dict = result[0]
                if isinstance(prediction_dict, dict) and 'classe' in prediction_dict:
                    classe = prediction_dict['classe']
                    return JSONResponse(content={"message": classe})

            elif isinstance(result, dict) and 'classe' in result:
                return JSONResponse(content={"message": result['classe']})

            else:
                result_str = str(result)
                if "'classe':" in result_str:
                    import re
                    match = re.search(r"'classe':\s*'([^']+)'", result_str)
                    if match:
                        return JSONResponse(content={"message": match.group(1)})

                return JSONResponse(content={"message": "Errore nell'estrazione della classe"})

        except Exception as e:
            print(f"Errore nell'estrazione della classe: {e}")
            return JSONResponse(content={"message": "Errore nell'elaborazione del risultato"})

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        print(f"Errore: {e}")
        raise HTTPException(status_code=500, detail=str(e))