import pandas as pd
from datetime import datetime
import os

OUTPUT_DIR = os.getenv("EXPORT_DIR", "./exports")


def export_to_excel(data: list[dict], filename: str = None, sheet_name: str = "Datos") -> str:
    """Exporta una lista de dicts a Excel y devuelve la ruta del fichero."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"football_report_{timestamp}.xlsx"

    filepath = os.path.join(OUTPUT_DIR, filename)
    df = pd.DataFrame(data)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

        # Autoajuste de columnas
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col) + 2
            ws.column_dimensions[col[0].column_letter].width = min(max_len, 40)

    return filepath


def export_multi_sheet(sheets: dict[str, list[dict]], filename: str = None) -> str:
    """Exporta múltiples datasets en distintas pestañas del mismo Excel."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"football_full_report_{timestamp}.xlsx"

    filepath = os.path.join(OUTPUT_DIR, filename)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for sheet_name, data in sheets.items():
            if data:
                df = pd.DataFrame(data)
                df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
                ws = writer.sheets[sheet_name[:31]]
                for col in ws.columns:
                    max_len = max(len(str(cell.value or "")) for cell in col) + 2
                    ws.column_dimensions[col[0].column_letter].width = min(max_len, 40)

    return filepath
