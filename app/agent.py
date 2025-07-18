# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import json
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

import google.auth
from google.adk.agents import Agent, RunConfig, LiveRequestQueue  # Importar Agent y RunConfig
from toolbox_core import ToolboxSyncClient # Importar MCP Toolbox for DBs de Google
from google.adk.runners import Runner
from google.genai import types as genai_types
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai.types import GenerateContentConfig

from google.cloud import bigquery
import uuid
import asyncio

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "europe-southwest1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

# --- Configuración del Modelo ---
MODEL_ID = "gemini-2.5-flash" # 

# --- Configuración de BigQuery ---
BIGQUERY_PROJECT_ID = "fon-test-project"
BIGQUERY_DATASET_ID = "foncorp_travel_data"
BIGQUERY_TABLE_ID = "travel_requests"

# --- Definición del Prompt ---
TRAVEL_AGENT_INSTRUCTION = f"""
Eres un amigable y eficiente asistente de viajes para los empleados de la empresa Foncorp.
Cuando un empleado inicie una conversación contigo, salúdalo cordialmente y preséntate indicando claramente qué puedes hacer por él en formato de lista.

Tus responsabilidades principales son:
- Registrar nuevas solicitudes de viaje.
- Consultar el estado de las solicitudes de viaje existentes.
- Modificar el estado de una solicitud de viaje específica a petición del usuario.
- Responder a consultas sobre las solicitudes de viaje ya registradas en la base de datos de foncorp travel.
- RESPOND IN SPANISH. YOU MUST RESPOND UNMISTAKABLY IN SPANISH, USING SPANISH ACCENT FROM SPAIN.

Estados Comunes de Solicitudes y sus Significados (para tu conocimiento interno y para interpretar consultas):
- 'Registrada': Solicitudes nuevas. Si el usuario pregunta por "pendientes", "nuevas", o "sin revisar", podría referirte a este estado o a una combinación con 'Pendiente de Aprobación'.
- 'Pendiente de Aprobación': Solicitudes revisadas y esperando decisión.
- 'Aprobada': Solicitudes aprobadas.
- 'Rechazada': Solicitudes no aprobadas.
- 'Reservada': Viajes reservados.
- 'Completada': Viajes ocurridos.
- 'Cancelada': Solicitudes canceladas.

Descripción del Esquema de la Tabla de Solicitudes de Viaje
A continuación se detalla el esquema de una tabla de base de datos que almacena solicitudes de viaje de empleados. 
Utiliza esta información para comprender la estructura de los datos y cómo interactuar con ellos.
Nombre de la Tabla: travel_requests (implícito)
Columnas de la Tabla:
request_id (STRING, OBLIGATORIO): Un identificador único para cada solicitud de viaje. Es la clave principal no oficial.
timestamp (TIMESTAMP, OBLIGATORIO): La fecha y hora exactas en que la solicitud fue creada o modificada por última vez.
employee_first_name (STRING, NULABLE): El nombre de pila del empleado que solicita el viaje.
employee_last_name (STRING, NULABLE): Los apellidos del empleado.
employee_id (STRING, NULABLE): El identificador único del empleado dentro de la empresa.
origin_city (STRING, NULABLE): La ciudad desde donde comienza el viaje.
destination_city (STRING, NULABLE): La ciudad a la que se dirige el empleado.
start_date (DATE, NULABLE): La fecha de inicio del viaje (formato AAAA-MM-DD).
end_date (DATE, NULABLE): La fecha de finalización del viaje (formato AAAA-MM-DD).
transport_mode (STRING, NULABLE): El medio de transporte preferido (ej. "Avión", "Tren", "Coche").
car_type (STRING, NULABLE): Si el transporte es "Coche", especifica si es "Particular" o de "Alquiler".
reason (STRING, NULABLE): Una breve descripción del motivo o propósito del viaje.
status (STRING, NULABLE): El estado actual de la solicitud (ej. "Registrada", "Aprobada", "Rechazada", "Cancelada").

Instrucciones para las herramientas:

1. Para registrar una nueva solicitud de viaje:
   - Recopila la siguiente información esencial: Nombre del empleado (pila), Apellidos del empleado, ID de empleado, Ciudad de Origen del viaje, Ciudad de Destino del viaje, Fecha de inicio (formato yyyy-MM-dd), Fecha de fin (formato yyyy-MM-dd), Medio de Transporte Preferido (Avión, Tren, Autobús, Coche), Tipo de Coche si aplica (Particular o Alquiler), y Motivo del viaje.
   - **Validación de Fechas Importante:**
     - Ambas fechas, inicio y fin, DEBEN ser futuras a la fecha actual ({datetime.datetime.now().strftime('%Y-%m-%d')}).
     - Si el usuario proporciona solo día y mes (ej. "15 de junio"), asume el año actual ({datetime.datetime.now().year}) para completar la fecha. Verifica que esta fecha resultante sea futura.
     - La fecha de fin no puede ser anterior a la fecha de inicio.
     - Si alguna fecha es inválida (pasada, o fin antes que inicio), NO llames a la herramienta. En su lugar, explica el problema al usuario y PÍDELE que proporcione fechas válidas. Por ejemplo: "Lo siento, la fecha [fecha inválida] ya ha pasado. Por favor, proporciona una fecha futura." o "La fecha de regreso no puede ser anterior a la de salida. Por favor, revisa las fechas."
   - Cuando tengas TODA la información válida (incluyendo fechas futuras y correctas), llama a la herramienta 'request_travel_booking_logic'.
   - Argumentos para 'request_travel_booking_logic': employee_first_name (str), employee_last_name (str), employee_id (str), origin_city (str), destination_city (str), start_date (str), end_date (str), transport_mode (str), reason (str), y opcionalmente car_type (str).

2. Para consultar solicitudes de viaje por estado:
   - **Proceso Estricto:** Cuando el usuario pida consultar solicitudes por estado (ej. 'Aprobada', 'Pendiente'):
     1. **Identifica el `search_term`** basado en la petición del usuario.
     2. **Llama INMEDIATAMENTE a la herramienta `get_travel_requests_by_status`** con este `search_term`.
     3. **NO GENERES NINGUNA RESPUESTA AL USUARIO ANTES DE RECIBIR EL RESULTADO DE LA HERRAMIENTA.** Espera la cadena JSON de la herramienta.
     4. **Una vez que la herramienta devuelva el JSON, analiza su contenido y USA ÚNICAMENTE ESE CONTENIDO para formular tu respuesta completa y final al usuario en este mismo turno.**
        - La herramienta devolverá datos como una cadena JSON: `{{"search_term": "...", "count": N, "requests": [{{"request_id": "...", ...}}], "message": "... opcional ..."}}` o `{{"message": "No se encontraron..."}}` o `{{"error": "..."}}`.
        - Si el JSON tiene `"count" > 0` y una lista de `"requests"`: Responde con algo como: "He encontrado [count] solicitudes [search_term]. Aquí están:
          - ID: [request_id_1], Empleado: [employee_name_1], Destino: [destination_city_1], Fechas: [start_date_1] a [end_date_1], Motivo: [reason_1]
          - ID: [request_id_2], Empleado: [employee_name_2], Destino: [destination_city_2], Fechas: [start_date_2] a [end_date_2], Motivo: [reason_2]
          (Continúa para todas las solicitudes en la lista `requests`)"
        - Si el JSON tiene un `"message"` (ej. no se encontraron resultados): Responde directamente con ese mensaje. Por ejemplo: "No se encontraron solicitudes para el término: [search_term]."
        - Si el JSON tiene un `"error"`: Responde informando del error. Por ejemplo: "Hubo un error al consultar las solicitudes: [error_message]."
     5. **ASEGÚRATE de que tu respuesta al usuario sea la presentación directa de los datos (o mensaje de no datos/error) recibidos de la herramienta, sin comentarios adicionales tuyos antes de presentar estos datos.**

3. Para actualizar el estado de una solicitud de viaje:
   - Necesitarás el ID de la solicitud ('request_id') y el nuevo estado ('new_status').
   - Pregunta al usuario por estos datos si no los proporciona. Asegúrate de que 'new_status' sea uno de los estados válidos listados arriba.
   - Llama a la herramienta 'update_travel_request_status' con los argumentos: request_id (str) y new_status (str).
   - Después de llamar a la herramienta, informa al usuario del resultado que devuelva la herramienta (confirmación o error).

4. Para cualquier otra consulta:
   - Utiliza la herramienta 'execute_sql_tool', con este table ID: fon-test-project.foncorp_travel_data.travel_requests.
   - Construye una consulta SQL en función de la información que suministre el cliente.
   - Ten en cuenta el esquema de la base de datos que se ha facilitado con estas instrucciones.

Reglas Generales:
- NO inventes información para las herramientas. Pide al usuario cualquier dato que falte.
- Sé siempre cortés y profesional.
- La fecha actual es: {datetime.datetime.now().strftime('%Y-%m-%d')}. Considera esto para inferir años si el usuario solo da día y mes para las fechas de viaje.
"""

# Conectamos con el Google MCP ToolBox Server (previamente hay que arrancarlo)
# toolbox = ToolboxSyncClient("http://mcp.fon.demo.altostrat.com:5000")
toolbox = ToolboxSyncClient("https://toolbox-429460911019.europe-southwest1.run.app")
#toolbox = ToolboxSyncClient("http://127.0.0.1:5000")
tools = toolbox.load_toolset('adk-travel-agent-toolset')

# --- (Opcional) Pydantic para claridad de argumentos ---
class _TravelBookingArgsSchema(BaseModel):
    employee_first_name: str = Field(description="Nombre del empleado (pila).")
    employee_last_name: str = Field(description="Apellidos del empleado.")
    employee_id: str = Field(description="ID del empleado.")
    origin_city: str = Field(description="Ciudad de origen del viaje.")
    destination_city: str = Field(description="Ciudad de destino del viaje.")
    start_date: str = Field(description="Fecha de inicio del viaje en formato yyyy-MM-dd.")
    end_date: str = Field(description="Fecha de fin del viaje en formato yyyy-MM-dd.")
    transport_mode: str = Field(description="Medio de transporte preferido.")
    reason: str = Field(description="Motivo del viaje.")
    car_type: Optional[str] = Field(default=None, description="Tipo de coche si es 'Coche' (Particular o Alquiler).")

class _GetTravelRequestsArgsSchema(BaseModel):
    search_term: str = Field(description="El estado o término de búsqueda para las solicitudes (ej. 'Cancelada', 'Pendiente', 'Registrada').")

class _UpdateTravelRequestArgsSchema(BaseModel):
    request_id: str = Field(description="ID de la solicitud a actualizar.")
    new_status: str = Field(description="Nuevo estado para la solicitud.")


# --- Lógica de la Herramienta 1: Registrar Solicitud (Usa DML INSERT) ---
def request_travel_booking_logic(
    employee_first_name: str,
    employee_last_name: str,
    employee_id: str,
    origin_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    transport_mode: str,
    reason: str,
    car_type: Optional[str] = None
) -> str:
    """Registra una solicitud de reserva de viaje en BigQuery con el nuevo esquema."""
    try:
        validated_args = _TravelBookingArgsSchema(
            employee_first_name=employee_first_name,
            employee_last_name=employee_last_name,
            employee_id=employee_id,
            origin_city=origin_city,
            destination_city=destination_city,
            start_date=start_date,
            end_date=end_date,
            transport_mode=transport_mode,
            reason=reason,
            car_type=car_type)
    except Exception as e:
        return f"Error de validación: {e}"
    try:
        date_format = "%Y-%m-%d"
        current_date_obj = datetime.datetime.now().date()
        start_date_obj = datetime.datetime.strptime(start_date, date_format).date()
        end_date_obj = datetime.datetime.strptime(end_date, date_format).date()

        if start_date_obj < current_date_obj:
            return f"Error en la herramienta: La fecha de inicio '{start_date}' ya ha pasado."
        if end_date_obj < current_date_obj:
             return f"Error en la herramienta: La fecha de fin '{end_date}' ya ha pasado."
        if end_date_obj < start_date_obj:
            return "Error en la herramienta: La fecha de fin no puede ser anterior a la fecha de inicio."
    except ValueError:
        return "Error en la herramienta: El formato de las fechas no es válido. Utiliza yyyy-MM-dd."

    try:
        client = bigquery.Client()
        request_id_val = str(uuid.uuid4())
        current_timestamp = datetime.datetime.now(datetime.timezone.utc)
        initial_status = "Registrada"
        table_ref_str = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_ID}"

        query = f"""
            INSERT INTO `{table_ref_str}` (
                request_id, timestamp, employee_first_name, employee_last_name, employee_id,
                origin_city, destination_city, start_date, end_date,
                transport_mode, car_type, reason, status
            ) VALUES (
                @request_id, @timestamp, @employee_first_name, @employee_last_name, @employee_id,
                @origin_city, @destination_city, @start_date, @end_date,
                @transport_mode, @car_type, @reason, @status
            )
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("request_id", "STRING", request_id_val),
                bigquery.ScalarQueryParameter("timestamp", "TIMESTAMP", current_timestamp.isoformat()),
                bigquery.ScalarQueryParameter("employee_first_name", "STRING", validated_args.employee_first_name),
                bigquery.ScalarQueryParameter("employee_last_name", "STRING", validated_args.employee_last_name),
                bigquery.ScalarQueryParameter("employee_id", "STRING", validated_args.employee_id),
                bigquery.ScalarQueryParameter("origin_city", "STRING", validated_args.origin_city),
                bigquery.ScalarQueryParameter("destination_city", "STRING", validated_args.destination_city),
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
                bigquery.ScalarQueryParameter("transport_mode", "STRING", transport_mode),
                bigquery.ScalarQueryParameter("car_type", "STRING", car_type),
                bigquery.ScalarQueryParameter("reason", "STRING", reason),
                bigquery.ScalarQueryParameter("status", "STRING", initial_status),
            ]
        )
        query_job = client.query(query, job_config=job_config)
        query_job.result()

        if query_job.errors:
            error_messages = "; ".join([str(error["message"]) for error in query_job.errors])
            print(f"[LOG request_travel_booking_logic - ERROR BQ DML]: {error_messages}")
            return f"Error al registrar la solicitud (DML): {error_messages}."
        else:
            if query_job.num_dml_affected_rows is not None and query_job.num_dml_affected_rows > 0:
                full_name = f"{validated_args.employee_first_name} {validated_args.employee_last_name}"
                confirmation_message = (
                    f"¡Solicitud registrada (DML)! ID: {request_id_val}. "
                    f"Para {full_name} (ID: {employee_id}) desde {origin_city} a {destination_city} "
                    f"({start_date} a {end_date}), usando {transport_mode}"
                    f"{f' ({car_type})' if car_type and transport_mode.lower() == 'coche' else ''}. Motivo: {reason}."
                )
                print(f"[LOG request_travel_booking_logic]: {confirmation_message}")
                return confirmation_message
            else:
                print(f"[LOG request_travel_booking_logic - ERROR BQ DML]: No se afectaron filas.")
                return "Error al registrar la solicitud: no se insertaron filas."
    except Exception as e:
        print(f"[LOG request_travel_booking_logic - ERROR]: {e}")
        return f"Error técnico al registrar la solicitud: {e}."

# --- Lógica de la Herramienta 2: Consultar Solicitudes por Estado (Devuelve JSON) ---
def get_travel_requests_by_status(search_term: str) -> str:
    """Consulta solicitudes de viaje por estado o término. Devuelve una cadena JSON."""
    try:
        validated_args = _GetTravelRequestsArgsSchema(
            search_term=search_term
        )
        search_term = validated_args.search_term
    except Exception as e:
        return json.dumps({"error": f"Error de validación: {e}"})
    try:
        client = bigquery.Client()
        table_ref_str = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_ID}"
        status_conditions = []
        query_params = []
        param_counter = 0
        processed_search_term = search_term.lower().strip()

        if "pendiente" in processed_search_term or \
           "sin aprobar" in processed_search_term or \
           "nuevas" in processed_search_term or \
           ("registrada" in processed_search_term and "aprobaci" not in processed_search_term) :
            param_counter += 1
            status_conditions.append(f"LOWER(status) = LOWER(@status_param_{param_counter})")
            query_params.append(bigquery.ScalarQueryParameter(f"status_param_{param_counter}", "STRING", "Registrada"))
            if "aprobaci" in processed_search_term or "pendiente" in processed_search_term :
                 param_counter += 1
                 if not any(p.value.lower() == "pendiente de aprobación" for p in query_params):
                    status_conditions.append(f"LOWER(status) = LOWER(@status_param_{param_counter})")
                    query_params.append(bigquery.ScalarQueryParameter(f"status_param_{param_counter}", "STRING", "Pendiente de Aprobación"))
        
        exact_final_statuses = ["aprobada", "rechazada", "reservada", "completada", "cancelada"]
        if processed_search_term in exact_final_statuses or \
           (not status_conditions and processed_search_term):
            capitalized_search = search_term.strip().capitalize()
            if capitalized_search in [s.capitalize() for s in exact_final_statuses]:
                 search_term_final = capitalized_search
            else:
                 search_term_final = search_term.strip()
            status_conditions = []
            query_params = []
            param_counter = 0
            param_counter += 1
            status_conditions.append(f"status = @status_param_{param_counter}")
            query_params.append(bigquery.ScalarQueryParameter(f"status_param_{param_counter}", "STRING", search_term_final))

        if not status_conditions:
             print(f"[LOG get_travel_requests_by_status]: Término no interpretado '{search_term}'.")
             return json.dumps({
                 "error": f"No pude interpretar el término de búsqueda de estado: '{search_term}'. Intenta usar uno de los estados conocidos (Registrada, Pendiente de Aprobación, Aprobada, Rechazada, Reservada, Completada, Cancelada)."
             })

        where_clause = " OR ".join(status_conditions)
        query = f"""
            SELECT request_id, employee_first_name, employee_last_name,
                   origin_city, destination_city, start_date, end_date, transport_mode, car_type, reason, status
            FROM `{table_ref_str}` WHERE {where_clause} ORDER BY timestamp DESC LIMIT 10
        """
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()

        if results.total_rows == 0:
            print(f"[LOG get_travel_requests_by_status]: No se encontraron solicitudes para '{search_term}'.")
            return json.dumps({
                "search_term": search_term,
                "count": 0,
                "requests": [],
                "message": f"No se encontraron solicitudes de viaje para el término: '{search_term}'."
            })

        output_requests = []
        for row in results:
            employee_full_name = f"{row.employee_first_name or ''} {row.employee_last_name or ''}".strip()
            request_data = {
                "request_id": str(row.request_id or "N/A"),
                "employee_name": str(employee_full_name or "N/A"),
                "origin_city": str(row.origin_city or "N/A"),
                "destination_city": str(row.destination_city or "N/A"),
                "start_date": str(row.start_date) if row.start_date else "N/A",
                "end_date": str(row.end_date) if row.end_date else "N/A",
                "transport_mode": str(row.transport_mode or "N/A"),
                "car_type": str(row.car_type or "N/A") if row.car_type else None,
                "reason": str(row.reason or "N/A"),
                "status": str(row.status or "N/A")
            }
            output_requests.append(request_data)
        
        print(f"[LOG DE HERRAMIENTA get_travel_requests_by_status]: JSON generado para '{search_term}'.")
        return json.dumps({
            "search_term": search_term,
            "count": results.total_rows,
            "requests": output_requests
        })

    except Exception as e:
        print(f"[LOG DE HERRAMIENTA get_travel_requests_by_status - ERROR]: {e}")
        return json.dumps({"error": f"Error técnico al consultar las solicitudes de viaje: {e}."})

# --- Lógica de la Herramienta 3: Actualizar Estado de Solicitud ---
def update_travel_request_status(request_id: str, new_status: str) -> str:
    """Actualiza el estado de una solicitud de viaje específica en BigQuery."""
    try:
        validated_args = _UpdateTravelRequestArgsSchema(
            request_id=request_id,
            new_status=new_status
        )
        request_id = validated_args.request_id
        new_status = validated_args.new_status
    except Exception as e:
        return f"Error de validación: {e}"
    valid_statuses = ["Registrada", "Pendiente de Aprobación", "Aprobada", "Rechazada", "Reservada", "Completada", "Cancelada"]
    
    status_map = {
        "registrada": "Registrada",
        "pendiente de aprobación": "Pendiente de Aprobación",
        "pendiente": "Pendiente de Aprobación",
        "aprobada": "Aprobada",
        "rechazada": "Rechazada",
        "reservada": "Reservada",
        "completada": "Completada",
        "cancelada": "Cancelada"
    }
    normalized_new_status = new_status.lower().strip()
    final_status = status_map.get(normalized_new_status)

    if not final_status:
        capitalized_status_direct = new_status.strip().capitalize()
        if capitalized_status_direct in valid_statuses:
             final_status = capitalized_status_direct
        else:
            return f"Error: '{new_status}' no es un estado válido. Válidos: {', '.join(valid_statuses)}."

    try:
        client = bigquery.Client()
        table_ref_str = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_ID}"
        query = f"""
            UPDATE `{table_ref_str}`
            SET status = @new_status_param, timestamp = @current_timestamp_param
            WHERE request_id = @request_id_param
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("new_status_param", "STRING", final_status),
                bigquery.ScalarQueryParameter("request_id_param", "STRING", request_id),
                bigquery.ScalarQueryParameter("current_timestamp_param", "TIMESTAMP", datetime.datetime.now(datetime.timezone.utc).isoformat())
            ]
        )
        query_job = client.query(query, job_config=job_config)
        query_job.result()

        if query_job.num_dml_affected_rows is not None and query_job.num_dml_affected_rows > 0:
            success_message = f"Solicitud ID '{request_id}' actualizada a '{final_status}'."
            print(f"[LOG update_travel_request_status]: {success_message}")
            return success_message
        else:
            check_query = f"SELECT status FROM `{table_ref_str}` WHERE request_id = @request_id_param"
            check_job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("request_id_param", "STRING", request_id)])
            check_job = client.query(check_query, job_config=check_job_config)
            check_results = list(check_job.result())
            if not check_results:
                 not_found_message = f"No se encontró solicitud con ID '{request_id}'."
            elif check_results[0].status == final_status:
                 not_found_message = f"La solicitud ID '{request_id}' ya estaba en estado '{final_status}'. No se realizaron cambios."
            else:
                 not_found_message = f"No se pudo actualizar la solicitud ID '{request_id}'. Razón desconocida."
            print(f"[LOG update_travel_request_status]: {not_found_message}")
            return not_found_message
    except Exception as e:
        error_message = f"Error técnico al actualizar estado de '{request_id}': {e}"
        print(f"[LOG update_travel_request_status - ERROR]: {error_message}")
        return error_message

# --- Creación del Agente y Configuración del RunConfig ---

# 1. Crear la instancia del Agente
root_agent = Agent(
    name="root_agent",
    description="Agente para gestionar solicitudes de viaje: registrar, consultar y actualizar estados.",
    instruction=TRAVEL_AGENT_INSTRUCTION,
    model=MODEL_ID,
    tools=[
        *tools,
        request_travel_booking_logic,
        get_travel_requests_by_status,
        update_travel_request_status
    ],
)