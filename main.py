"""
Football Analytics Agent
========================
Modos de uso:
  1. CLI interactivo : python main.py
  2. API REST        : uvicorn main:app --reload  →  http://localhost:8000/docs
"""

import os
import sys
from pprint import pprint
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Annotated
import db

from agent import MENU_CASES, AVAILABLE_STATS, detect_intent, extract_year, extract_stat, run_case
from exporter import export_to_excel, export_multi_sheet

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["RUNTIME"] = "api"  # solo se ejecuta con uvicorn
    yield
    os.environ.pop("RUNTIME", None)  # limpieza al parar

app = FastAPI(title="Football Analytics Agent", version="1.0", lifespan=lifespan)


def make_enum(name: str, values: list[str]) -> Enum:
    return Enum(name, {v: v for v in values})

TeamEnum = make_enum("TeamEnum", db.get_teams())
YearEnum = make_enum("YearEnum", db.get_years())
StatEnum = make_enum("StatEnum", db.get_stats())


# ─────────────────────────────────────────────
# MODELOS PYDANTIC
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str
    team: Optional[str] = None
    league: Optional[str] = None
    year: Optional[int] = None
    stat: Optional[str] = None
    export_excel: bool = False


class MenuRequest(BaseModel):
    case_id: str          # "1" .. "6"
    team: Annotated[Optional[str], Field(description="Equipo")] = None
    year: Annotated[Optional[str], Field(description="Año")] = None
    league: Optional[str] = None
    stat: Annotated[Optional[str], Field(description="Stat")] = None
    top_n: int = 10
    export_excel: bool = False


# ─────────────────────────────────────────────
# ENDPOINTS API
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Football Analytics Agent activo ✅", "docs": "/docs"}


@app.get("/menu")
def get_menu():
    """Devuelve los casos predefinidos disponibles."""
    return {"cases": MENU_CASES, "available_stats": AVAILABLE_STATS}


@app.get("/query/results")
def query_results(
    team: TeamEnum = Query(description="Selecciona el equipo"),
    year: YearEnum = Query(description="Selecciona el año"),
    stat: StatEnum = Query(description="Selecciona la stat"),
    limit: Optional[int] = None,
    export_excel: bool = False
):
    result = run_case(handler="results", team=team.value, year=year.value, stat=stat.value, top_n=limit)
    return result


@app.post("/query/menu")
def query_by_menu(req: MenuRequest):
    """Ejecuta un caso predefinido del menú."""
    case = MENU_CASES.get(req.case_id)
    if not case:
        raise HTTPException(status_code=400, detail=f"Caso '{req.case_id}' no existe. Usa GET /menu")

    result = run_case(
        handler=case["handler"],
        team=req.team,
        league=req.league,
        year=req.year,
        stat=req.stat,
        top_n=req.top_n
    )

    response = {"case": case["label"], "output": result["text"]}

    if req.export_excel:
        data = result["data"]
        if isinstance(data, dict):
            path = export_multi_sheet(data)
        else:
            path = export_to_excel(data, sheet_name=case["label"])
        response["excel"] = path

    return response


# @app.post("/query/text")
def query_by_text(req: QueryRequest):
    """Interpreta lenguaje natural y ejecuta la consulta más probable."""
    intent = detect_intent(req.text)
    if not intent:
        raise HTTPException(
            status_code=422,
            detail="No se pudo interpretar la consulta. Prueba con palabras como: resultados, goles, clasificación, posesión..."
        )

    year = req.year or extract_year(req.text)
    stat = req.stat or extract_stat(req.text)

    result = run_case(
        handler=intent,
        team=req.team,
        league=req.league,
        year=year,
        stat=stat
    )

    response = {"intent_detected": intent, "output": result["text"]}

    if req.export_excel:
        data = result["data"]
        if isinstance(data, dict):
            path = export_multi_sheet(data)
        else:
            path = export_to_excel(data)
        response["excel"] = path

    return response


# @app.get("/export/download")
def download_file(path: str):
    """Descarga un fichero Excel generado previamente."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Fichero no encontrado")
    return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename=os.path.basename(path))


# ─────────────────────────────────────────────
# MODO CLI INTERACTIVO
# ─────────────────────────────────────────────

def cli_ask(prompt: str, default: str = None) -> str:
    val = input(f"  {prompt}{f' [{default}]' if default else ''}: ").strip()
    return val or default or ""


def cli_menu():
    print("\n" + "═"*55)
    print("  🏟️  FOOTBALL ANALYTICS AGENT")
    print("═"*55)
    print("\n  Escribe tu consulta en lenguaje natural")
    print("  o elige un caso predefinido:\n")
    for k, v in MENU_CASES.items():
        print(f"    [{k}] {v['label']}")
    print("    [0] Salir")
    print()


def cli_collect_params(case: dict) -> dict:
    params = {}
    for p in case["params"]:
        optional = p.endswith("?")
        name = p.rstrip("?")
        label = f"{name} (opcional)" if optional else name
        val = cli_ask(label)
        if val:
            params[name] = int(val) if name in ("top_n") else val
    return params


def run_cli():
    while True:
        cli_menu()
        choice = input("  Tu elección o consulta: ").strip()

        if choice == "0":
            print("\n  ¡Hasta luego! ⚽\n")
            sys.exit(0)

        elif choice in MENU_CASES:
            case = MENU_CASES[choice]
            print(f"\n  → {case['label']}")
            params = cli_collect_params(case)
            result = run_case(handler=case["handler"], **params)
            print(result["text"])

        else:
            # Modo texto libre
            intent = detect_intent(choice)
            if not intent:
                print("\n  ⚠️  No entendí la consulta. Intenta con: resultados, goles, clasificación...\n")
                continue

            team   = cli_ask("Equipo") if intent not in ("standings", "top_stats") else None
            league = cli_ask("Liga (opcional)") or None
            year   = cli_ask("Año (opcional)") or None
            stat   = extract_stat(choice) or (cli_ask(f"Estadística {AVAILABLE_STATS}") if intent in ("stat", "top_stats") else None)

            result = run_case(
                handler=intent, team=team,
                league=league, year=int(year) if year else None, stat=stat
            )
            pprint(result["text"])

        export = input("  ¿Exportar a Excel? (s/N): ").strip().lower()
        if export == "s":
            data = result["data"]
            if isinstance(data, dict):
                path = export_multi_sheet(data)
            else:
                path = export_to_excel(data)
            print(f"  ✅ Excel guardado en: {path}\n")


if __name__ == "__main__":
    run_cli()
