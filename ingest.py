import os
import glob
import shutil
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

# 1. Limpiar base de datos previa
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("🧹 Base de datos antigua eliminada.")

# 2. Buscar automáticamente el archivo Excel
archivos_excel = glob.glob("*.xlsx")

if not archivos_excel:
    print("❌ ERROR: No se encontró ningún archivo .xlsx en la carpeta.")
    exit()

excel_file = archivos_excel[0]
print(f"📖 Cargando datos desde el archivo: {excel_file}")

# 3. Leer pestañas del Excel
excel_completo = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')

documentos = []

for nombre_hoja, df in excel_completo.items():
    df = df.fillna('')
    for indice, fila in df.iterrows():
        lineas = [f"Categoría: {nombre_hoja}"]
        
        for columna, valor in fila.items():
            col_name = str(columna).strip()
            if "Unnamed" in col_name or not col_name:
                continue
                
            val_str = str(valor).strip()
            if val_str != '':
                lineas.append(f"{col_name}: {val_str}")
        
        contenido = "\n".join(lineas)
        
        doc = Document(
            page_content=contenido,
            metadata={"categoria": nombre_hoja, "fila": indice + 1}
        )
        documentos.append(doc)

# 4. Embeddings ultraligeros sin dependencia de torch/DLLs
print("⏳ Generando embeddings con FastEmbed...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 5. Guardar en ChromaDB
vectorstore = Chroma.from_documents(
    documents=documentos,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"✅ ¡Éxito! Se actualizaron {len(documentos)} productos en ./chroma_db")