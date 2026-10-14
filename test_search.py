import re
import os
from datetime import datetime
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# 1. Cargar el modelo de embeddings liviano (FastEmbed)
print("⏳ Cargando modelo de embeddings FastEmbed...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# 2. Cargar la base de datos vectorial local
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Memoria local
ultimas_opciones = {}
carrito = []

def limpiar_html(texto):
    """Elimina etiquetas HTML de un texto."""
    if not isinstance(texto, str):
        return str(texto)
    return re.sub(r'<[^>]+>', '', texto).strip()

def extraer_precio_numerico(texto_doc):
    """Extrae el precio adaptado al nuevo formato (busca en la línea Variant Price o precios)."""
    for line in texto_doc.split('\n'):
        if "variant price" in line.lower() or line.lower().startswith("precio"):
            numeros = re.findall(r'\d+', line.replace('.', ''))
            if numeros:
                return int(numeros[-1])
    # Fallback: buscar cualquier número grande que parezca precio al final del documento
    numeros_totales = re.findall(r'\d+', texto_doc.replace('.', ''))
    if numeros_totales:
        # Por lo general el precio es el último número del documento en este formato
        posible_precio = int(numeros_totales[-1])
        if posible_precio > 99: # Evita tomar índices pequeños
            return posible_precio
    return 0

def extraer_nombre_producto(texto_doc):
    """Extrae el título adaptado al nuevo formato (columna Title)."""
    for line in texto_doc.split('\n'):
        if line.lower().startswith("title") or "title:" in line.lower():
            parts = line.split(":", 1)
            if len(parts) > 1:
                return limpiar_html(parts[1])
    
    # Si no tiene la etiqueta Title explícita, tomamos la primera línea útil
    lineas = [l.strip() for l in texto_doc.split('\n') if l.strip()]
    if lineas:
        return limpiar_html(lineas[0])
    return "Producto sin nombre"

def generar_pdf_cotizacion(items_carrito):
    """Genera un archivo PDF profesional con la cotización acumulada."""
    nombre_archivo = f"Cotizacion_PitsDetailing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(nombre_archivo, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    titulo_style = ParagraphStyle(
        'TituloStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )
    
    elements.append(Paragraph("PITS DETAILING - COTIZACIÓN DE PRODUCTOS", titulo_style))
    elements.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 15))
    
    data = [["N°", "Producto / Descripción", "Precio Unitario"]]
    total = 0
    for idx, item in enumerate(items_carrito, 1):
        precio_fmt = f"${item['precio']:,}".replace(',', '.')
        data.append([str(idx), item['nombre'], precio_fmt])
        total += item['precio']
        
    data.append(["", "TOTAL COTIZADO", f"${total:,}".replace(',', '.')])
    
    tabla = Table(data, colWidths=[30, 340, 130])
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -2), 0.5, colors.HexColor("#E2E8F0")),
        ('LINEABOVE', (0, -1), (-1, -1), 1.5, colors.HexColor("#2B6CB0")),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    
    elements.append(tabla)
    doc.build(elements)
    return nombre_archivo

def buscar_con_prioridad(pregunta, k=3):
    """Realiza una búsqueda híbrida priorizando coincidencias exactas de términos clave."""
    # 1. Obtenemos un poco más de resultados del vectorstore para filtrar
    resultados_raw = vectorstore.similarity_search(pregunta, k=10)
    
    palabras_clave = [p.lower() for p in pregunta.split() if len(p) > 2]
    resultados_filtrados = []
    
    # 2. Priorizar documentos que contengan palabras clave importantes (ej: 'shampoo', 'limpiador')
    for doc in resultados_raw:
        texto_lower = doc.page_content.lower()
        # Si el usuario busca algo muy específico como shampoo, filtramos para que tenga sentido
        coincidencias = sum(1 for palabra in palabras_clave if palabra in texto_lower)
        
        # Guardamos el documento junto con su puntaje de coincidencia de palabras
        resultados_filtrados.append((coincidencias, doc))
        
    # 3. Ordenar por mayor cantidad de coincidencias de palabras clave exactas
    resultados_filtrados.sort(key=lambda x: x[0], reverse=True)
    
    # Devolver los top `k` mejores
    mejores_docs = [doc for _, doc in resultados_filtrados[:k]]
    
    # Si por alguna razón el filtro strict deja vacío, devolvemos los raw normales
    if not mejores_docs:
        return resultados_raw[:k]
        
    return mejores_docs


# 3. Bucle interactivo inteligente con soporte multiselección
if __name__ == "__main__":
    print("🤖 ¡Asistente Pits Detailing listo!\n")
    print("Instrucciones:")
    print(" - Escribe lo que buscas para ver opciones.")
    print(" - Selecciona varios productos escribiendo sus números (ej: 1, 2, 3).")
    print(" - Escribe 'cotizar' para generar el PDF con tu carrito.")
    print(" - Escribe 'salir' para terminar.\n")
    
    while True:
        pregunta = input("Tú: ").strip()
        
        if not pregunta:
            continue
            
        preq_lower = pregunta.lower()
        
        if preq_lower in ["salir", "exit"]:
            print("¡Hasta luego!")
            break
            
        # Opción A: Generar cotización en PDF
        elif "cotizar" in preq_lower or "cotizacion" in preq_lower or "comprar" in preq_lower:
            if not carrito:
                print("⚠️ Asistente: Tu carrito está vacío. Agrega productos escribiendo sus números antes de cotizar.\n")
            else:
                archivo = generar_pdf_cotizacion(carrito)
                print(f"✅ ¡Cotización generada con éxito! Archivo guardado como: {archivo}\n")
                carrito.clear()
            continue
            
        # Opción B: Detectar si el usuario escribió números para agregarlos al carrito
        numeros_encontrados = [int(n) for n in re.findall(r'\d+', pregunta)]
        
        if numeros_encontrados and (len(pregunta) < 30 or any(p in preq_lower for p in ["la", "opcion", "el", "item", "quiero", "agrega", "agregar", ",", " y "])):
            agregados = 0
            for num in numeros_encontrados:
                if num in ultimas_opciones:
                    producto_elegido = ultimas_opciones[num]
                    carrito.append(producto_elegido)
                    print(f"🛒 ¡Agregado al carrito!: {producto_elegido['nombre']} (${producto_elegido['precio']:,})".replace(',', '.'))
                    agregados += 1
                else:
                    print(f"❌ El número {num} no está en la última lista mostrada.")
            
            if agregados > 0:
                print(f"\n📦 Productos totales en el carrito: {len(carrito)}. Escribe 'cotizar' cuando quieras generar tu PDF.\n")
            continue
            
        # Opción C: Búsqueda normal por texto
        else:
            docs = buscar_con_prioridad(pregunta, k=3)
            
            if not docs:
                print("Asistente: No encontré productos relacionados con tu consulta.\n")
                continue
                
            print("\nAsistente: Encontré las siguientes opciones:")
            ultimas_opciones.clear()
            
            for idx, doc in enumerate(docs, 1):
                nombre = extraer_nombre_producto(doc.page_content)
                precio = extraer_precio_numerico(doc.page_content)
                ultimas_opciones[idx] = {"nombre": nombre, "precio": precio}
                print(f"  [{idx}] {nombre} - ${precio:,}".replace(',', '.'))
            
            print("\n👉 Escribe los números de las opciones (ej: 1, 2) para agregarlos, o escribe otra búsqueda.\n")