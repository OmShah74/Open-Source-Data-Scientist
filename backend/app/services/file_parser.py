import pandas as pd
from fastapi import UploadFile
from io import BytesIO
import docx

async def parse_file(file: UploadFile):
    """
    Parses an uploaded file (CSV, Excel, or DOCX with a table) into a pandas DataFrame.
    """
    content = await file.read()
    if file.filename.endswith('.csv'):
        return pd.read_csv(BytesIO(content))
    elif file.filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(BytesIO(content))
    elif file.filename.endswith('.docx'):
        doc = docx.Document(BytesIO(content))
        # This is a simplified parser assuming the first table is the data
        table = doc.tables[0]
        data = []
        keys = [cell.text for cell in table.rows[0].cells]
        for i, row in enumerate(table.rows):
            if i == 0:
                continue
            row_data = {keys[j]: cell.text for j, cell in enumerate(row.cells)}
            data.append(row_data)
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format.")