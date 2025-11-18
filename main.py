from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import random

app = FastAPI()

# Configurar CORS para permitir requests desde GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tu dominio de GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELOS ====================
class Arista(BaseModel):
    origen: int
    destino: int
    capacidad: int

class GrafoInput(BaseModel):
    nodos: List[int]
    aristas: List[Arista]
    fuente: int
    sumidero: int

class GenerarGrafoInput(BaseModel):
    n_nodos: int
    fuente: Optional[int] = None
    sumidero: Optional[int] = None

# ==================== CLASES ORIGINALES ====================
class GrafoDirigido:
    def __init__(self):
        self.grafo = defaultdict(dict)
        self.nodos = set()
        self.aristas_info = []
    
    def agregar_arista(self, origen, destino, capacidad):
        for i, arista in enumerate(self.aristas_info):
            if arista['origen'] == origen and arista['destino'] == destino:
                self.aristas_info[i]['capacidad'] = capacidad
                self.aristas_info[i]['etiqueta'] = f"{origen}→{destino}: {capacidad}"
                self.grafo[origen][destino] = capacidad
                return
        
        self.grafo[origen][destino] = capacidad
        self.nodos.add(origen)
        self.nodos.add(destino)
        self.aristas_info.append({
            'origen': origen,
            'destino': destino, 
            'capacidad': capacidad,
            'etiqueta': f"{origen}→{destino}: {capacidad}"
        })
    
    def obtener_capacidad(self, origen, destino):
        return self.grafo[origen].get(destino, 0)

class FordFulkerson:
    def __init__(self, grafo):
        self.grafo_original = grafo
        self.grafo_residual = self.crear_grafo_residual()
        self.flujo_resultado = defaultdict(lambda: defaultdict(int))
        self.caminos_encontrados = []
        self.etiquetas_por_paso = []
        self.detallado_completo = []
    
    def crear_grafo_residual(self):
        residual = defaultdict(dict)
        for origen in self.grafo_original.grafo:
            for destino, capacidad in self.grafo_original.grafo[origen].items():
                residual[origen][destino] = capacidad
                if destino not in residual or origen not in residual[destino]:
                    residual[destino][origen] = 0
        return residual
    
    def bfs_camino_aumentante_con_etiquetas(self, fuente, sumidero):
        etiquetas = {}
        visitado = set()
        cola = deque([fuente])
        visitado.add(fuente)
        
        etiquetas[fuente] = ('-', float('inf'), None)
        
        log = f"  Etiquetado de nodos:\n"
        log += f"    Nodo {fuente}: (-, ∞)\n"
        
        while cola:
            nodo_actual = cola.popleft()
            delta_actual = etiquetas[nodo_actual][1]
            
            for vecino, capacidad_residual in self.grafo_residual[nodo_actual].items():
                if vecino not in visitado and capacidad_residual > 0:
                    es_directa = self.grafo_original.obtener_capacidad(nodo_actual, vecino) > 0
                    
                    if es_directa:
                        delta_nuevo = min(delta_actual, capacidad_residual)
                        etiquetas[vecino] = (nodo_actual, delta_nuevo, '+')
                        log += f"    Nodo {vecino}: ({nodo_actual}+, {delta_nuevo})\n"
                    else:
                        flujo_actual = self.flujo_resultado[vecino][nodo_actual]
                        if flujo_actual > 0:
                            delta_nuevo = min(delta_actual, flujo_actual)
                            etiquetas[vecino] = (nodo_actual, delta_nuevo, '-')
                            log += f"    Nodo {vecino}: ({nodo_actual}-, {delta_nuevo})\n"
                    
                    visitado.add(vecino)
                    cola.append(vecino)
                    
                    if vecino == sumidero:
                        return self.reconstruir_camino(etiquetas, fuente, sumidero), etiquetas, log
        
        return None, etiquetas, log
    
    def reconstruir_camino(self, etiquetas, fuente, sumidero):
        camino = []
        nodo_actual = sumidero
        
        while nodo_actual != fuente:
            camino.append(nodo_actual)
            nodo_actual = etiquetas[nodo_actual][0]
        
        camino.append(fuente)
        return camino[::-1]
    
    def calcular_flujo_maximo(self, fuente, sumidero):
        flujo_maximo = 0
        paso = 0
        
        output = "=" * 70 + "\n"
        output += "ALGORITMO FORD-FULKERSON CON ETIQUETADO\n"
        output += "=" * 70 + "\n"
        output += f"Fuente: {fuente}, Sumidero: {sumidero}\n\n"
        
        while True:
            paso += 1
            output += f"\n{'─' * 70}\n"
            output += f"PASO {paso}: Búsqueda de camino aumentante\n"
            output += f"{'─' * 70}\n"
            
            camino_tuple = self.bfs_camino_aumentante_con_etiquetas(fuente, sumidero)
            camino = camino_tuple[0]
            etiquetas = camino_tuple[1]
            log_etiquetado = camino_tuple[2]
            
            output += log_etiquetado
            
            if not camino:
                output += f"\n  No se encontró camino aumentante\n"
                output += f"  El sumidero {sumidero} no fue etiquetado\n"
                break
            
            self.etiquetas_por_paso.append(etiquetas.copy())
            
            output += f"\n  Camino aumentante encontrado: {' → '.join(map(str, camino))}\n"
            
            capacidad_minima = etiquetas[sumidero][1]
            
            output += f"\n  Análisis de aristas en el camino:\n"
            for i in range(len(camino) - 1):
                origen, destino = camino[i], camino[i + 1]
                capacidad_res = self.grafo_residual[origen][destino]
                capacidad_orig = self.grafo_original.obtener_capacidad(origen, destino)
                flujo_act = self.flujo_resultado[origen][destino]
                
                if capacidad_orig > 0:
                    output += f"    {origen} → {destino}: capacidad residual = {capacidad_res}, "
                    output += f"capacidad original = {capacidad_orig}, flujo actual = {flujo_act}\n"
                else:
                    output += f"    {origen} → {destino}: arista inversa (reduciendo flujo {destino}→{origen})\n"
            
            output += f"\n  Δ(sumidero) = {capacidad_minima} ← Flujo a enviar por el camino\n"
            
            for i in range(len(camino) - 1):
                origen, destino = camino[i], camino[i + 1]
                self.grafo_residual[origen][destino] -= capacidad_minima
                self.grafo_residual[destino][origen] += capacidad_minima
                
                if destino in self.grafo_original.grafo[origen]:
                    self.flujo_resultado[origen][destino] += capacidad_minima
                else:
                    self.flujo_resultado[destino][origen] -= capacidad_minima
            
            flujo_maximo += capacidad_minima
            self.caminos_encontrados.append((camino.copy(), capacidad_minima))
            
            output += f"\n  Flujo acumulado total: {flujo_maximo}\n"
        
        output += f"\n{'=' * 70}\n"
        output += f"RESULTADO FINAL\n"
        output += f"{'=' * 70}\n"
        output += f"Flujo máximo encontrado: {flujo_maximo}\n\n"
        
        output += "Asignación de flujo por arista:\n"
        for origen in sorted(self.flujo_resultado.keys()):
            for destino in sorted(self.flujo_resultado[origen].keys()):
                flujo = self.flujo_resultado[origen][destino]
                if flujo > 0:
                    capacidad_original = self.grafo_original.obtener_capacidad(origen, destino)
                    output += f"  {origen} → {destino}: {flujo}/{capacidad_original}\n"
        
        self.detallado_completo = output
        return flujo_maximo
    
    def obtener_corte_minimo(self, fuente, sumidero):
        visitado = set()
        cola = deque([fuente])
        visitado.add(fuente)
        
        while cola:
            nodo = cola.popleft()
            for vecino, capacidad in self.grafo_residual[nodo].items():
                if vecino not in visitado and capacidad > 0:
                    visitado.add(vecino)
                    cola.append(vecino)
        
        corte_minimo = []
        capacidad_corte = 0
        
        for origen in visitado:
            for destino in self.grafo_original.nodos:
                if destino not in visitado:
                    cap = self.grafo_original.obtener_capacidad(origen, destino)
                    if cap > 0:
                        corte_minimo.append((origen, destino))
                        capacidad_corte += cap
        
        return corte_minimo, list(visitado), capacidad_corte
    
    def buscar_soluciones_alternativas(self, fuente, sumidero, flujo_maximo):
        output = f"\n{'=' * 70}\n"
        output += f"BÚSQUEDA DE SOLUCIONES ALTERNATIVAS\n"
        output += f"{'=' * 70}\n"
        
        aristas_no_saturadas = []
        for origen in self.flujo_resultado:
            for destino, flujo in self.flujo_resultado[origen].items():
                if flujo > 0:
                    capacidad = self.grafo_original.obtener_capacidad(origen, destino)
                    if flujo < capacidad:
                        aristas_no_saturadas.append((origen, destino, flujo, capacidad))
        
        if aristas_no_saturadas:
            output += f"\nAristas con capacidad no saturada:\n"
            for origen, destino, flujo, capacidad in aristas_no_saturadas:
                output += f"  {origen} → {destino}: {flujo}/{capacidad} (disponible: {capacidad - flujo})\n"
            output += f"\nExisten posibles redistribuciones del flujo manteniendo el valor máximo {flujo_maximo}\n"
        else:
            output += f"\nTodas las aristas del flujo están saturadas.\n"
            output += f"La solución encontrada es única (no hay redistribuciones posibles).\n"
        
        return output, aristas_no_saturadas

# ==================== ENDPOINTS ====================

@app.get("/")
def read_root():
    return {"message": "Ford-Fulkerson API - Backend funcionando correctamente"}

@app.post("/generar-grafo")
def generar_grafo(data: GenerarGrafoInput):
    n = data.n_nodos
    if n < 8 or n > 16:
        return {"error": "El número de nodos debe estar entre 8 y 16"}
    
    grafo = GrafoDirigido()
    nodos = list(range(n))
    
    for nodo in nodos:
        grafo.nodos.add(nodo)
    
    fuente = data.fuente if data.fuente is not None else 0
    sumidero = data.sumidero if data.sumidero is not None else n - 1
    
    if fuente == sumidero:
        sumidero = n - 1 if fuente == 0 else 0
    
    intermedios = [x for x in nodos if x != fuente and x != sumidero]
    random.shuffle(intermedios)
    
    num_intermedios = len(intermedios)
    base = num_intermedios // 3
    resto = num_intermedios % 3
    tamanos = [base, base, base]
    for i in range(resto):
        tamanos[i] += 1
    
    idx = 0
    capa1 = sorted(intermedios[idx: idx + tamanos[0]]); idx += tamanos[0]
    capa2 = sorted(intermedios[idx: idx + tamanos[1]]); idx += tamanos[1]
    capa3 = sorted(intermedios[idx: idx + tamanos[2]])
    
    # Conectar fuente a capa1
    if capa1:
        for destino in capa1:
            capacidad = random.randint(8, 20)
            grafo.agregar_arista(fuente, destino, capacidad)
    elif capa2:
        for destino in capa2:
            capacidad = random.randint(8, 20)
            grafo.agregar_arista(fuente, destino, capacidad)
    elif capa3:
        for destino in capa3:
            capacidad = random.randint(8, 20)
            grafo.agregar_arista(fuente, destino, capacidad)
    else:
        capacidad = random.randint(15, 25)
        grafo.agregar_arista(fuente, sumidero, capacidad)
    
    # Conectar capa1 a capa2
    for nodo in capa1:
        if capa2:
            num_conexiones = random.randint(1, min(2, len(capa2)))
            destinos = random.sample(capa2, num_conexiones)
            for dest in destinos:
                capacidad = random.randint(5, 18)
                grafo.agregar_arista(nodo, dest, capacidad)
    
    # Conectar capa2 a capa3
    for nodo in capa2:
        if capa3:
            num_conexiones = random.randint(1, min(2, len(capa3)))
            destinos = random.sample(capa3, num_conexiones)
            for dest in destinos:
                capacidad = random.randint(5, 18)
                grafo.agregar_arista(nodo, dest, capacidad)
    
    # Conectar capa3 a sumidero
    for nodo in capa3:
        capacidad = random.randint(8, 20)
        grafo.agregar_arista(nodo, sumidero, capacidad)
    
    return {
        "nodos": list(grafo.nodos),
        "aristas": grafo.aristas_info,
        "fuente": fuente,
        "sumidero": sumidero
    }

@app.post("/calcular-flujo")
def calcular_flujo(data: GrafoInput):
    grafo = GrafoDirigido()
    
    # Construir grafo
    for nodo in data.nodos:
        grafo.nodos.add(nodo)
    
    for arista in data.aristas:
        grafo.agregar_arista(arista.origen, arista.destino, arista.capacidad)
    
    # Ejecutar Ford-Fulkerson
    ff = FordFulkerson(grafo)
    flujo_maximo = ff.calcular_flujo_maximo(data.fuente, data.sumidero)
    
    # Obtener corte mínimo
    corte_minimo, nodos_visitados, capacidad_corte = ff.obtener_corte_minimo(data.fuente, data.sumidero)
    
    # Buscar soluciones alternativas
    output_alternativas, aristas_no_saturadas = ff.buscar_soluciones_alternativas(
        data.fuente, data.sumidero, flujo_maximo
    )
    
    # Preparar flujos por arista
    flujos = []
    for origen in ff.flujo_resultado:
        for destino, flujo in ff.flujo_resultado[origen].items():
            if flujo > 0:
                capacidad = grafo.obtener_capacidad(origen, destino)
                flujos.append({
                    "origen": origen,
                    "destino": destino,
                    "flujo": flujo,
                    "capacidad": capacidad
                })
    
    detallado_completo = ff.detallado_completo
    detallado_completo += f"\n{'=' * 70}\n"
    detallado_completo += f"CORTE MÍNIMO (VALIDACIÓN DEL RESULTADO)\n"
    detallado_completo += f"{'=' * 70}\n"
    detallado_completo += f"\nConjunto S (alcanzables desde fuente): {sorted(nodos_visitados)}\n"
    detallado_completo += f"Conjunto T (no alcanzables): {sorted([n for n in grafo.nodos if n not in nodos_visitados])}\n"
    detallado_completo += f"\nAristas del corte mínimo (S → T):\n"
    
    for origen, destino in corte_minimo:
        capacidad = grafo.obtener_capacidad(origen, destino)
        detallado_completo += f"  {origen} → {destino}: capacidad = {capacidad}\n"
    
    detallado_completo += f"\n{'─' * 70}\n"
    detallado_completo += f"Capacidad total del corte = {capacidad_corte}\n"
    detallado_completo += f"Flujo máximo calculado = {flujo_maximo}\n"
    detallado_completo += f"{'─' * 70}\n"
    
    if flujo_maximo == capacidad_corte:
        detallado_completo += f"\n✅ VERIFICACIÓN CORRECTA:\n"
        detallado_completo += f"   Flujo máximo ({flujo_maximo}) = Capacidad corte mínimo ({capacidad_corte})\n"
        detallado_completo += f"   El resultado está VALIDADO por el teorema de flujo máximo-corte mínimo\n"
    else:
        detallado_completo += f"\n⚠️ ADVERTENCIA:\n"
        detallado_completo += f"   Flujo máximo ({flujo_maximo}) ≠ Capacidad corte mínimo ({capacidad_corte})\n"
        detallado_completo += f"   Puede haber un error en el cálculo\n"
    
    detallado_completo += output_alternativas
    
    return {
        "flujo_maximo": flujo_maximo,
        "flujos": flujos,
        "caminos": [{"camino": camino, "flujo": flujo} for camino, flujo in ff.caminos_encontrados],
        "corte_minimo": {
            "aristas": [{"origen": o, "destino": d} for o, d in corte_minimo],
            "capacidad": capacidad_corte,
            "conjunto_s": nodos_visitados,
            "conjunto_t": [n for n in grafo.nodos if n not in nodos_visitados]
        },
        "detallado_completo": detallado_completo,
        "validado": flujo_maximo == capacidad_corte
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
