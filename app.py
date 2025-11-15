from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os, json, datetime, secrets
from pathlib import Path
from contextlib import asynccontextmanager

from agent import LumenAgent

# ======================================================
# INICIALIZAÇÃO DO FASTAPI
# ======================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔥 Lúmen iniciada.")
    yield
    print("👋 Encerrando…")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Pastas
os.makedirs("data/users", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

AGENT = LumenAgent()

# ======================================================
# MODELOS
# ======================================================

class LoginData(BaseModel):
    username: str
    password: str

class TokenData(BaseModel):
    username: str
    token: str

class ChatRequest(BaseModel):
    username: str
    token: str
    chat_id: str
    message: str = ""

class NovaConversaData(BaseModel):
    username: str
    token: str

class RenomearData(BaseModel):
    username: str
    token: str
    chat_id: str
    novo_nome: str

# ======================================================
# FUNÇÕES INTERNAS
# ======================================================

def user_path(username):
    return f"data/users/{username}.json"

def load_user(username):
    path = user_path(username)
    if not os.path.exists(path):
        return None
    return json.load(open(path, "r", encoding="utf-8"))

def save_user(username, data):
    json.dump(data, open(user_path(username), "w", encoding="utf-8"), indent=4)

def verificar_sessao(username, token):
    user = load_user(username)
    if not user:
        raise HTTPException(404, "Usuário não encontrado.")
    if user.get("token") != token:
        raise HTTPException(403, "Sessão expirada. Faça login novamente.")
    return user

# ======================================================
# LOGIN / SESSÃO
# ======================================================

@app.post("/api/login")
def login(data: LoginData):
    upath = user_path(data.username)

    if not os.path.exists(upath):
        token = secrets.token_hex(32)
        novo = {
            "password": data.password,
            "token": token,
            "conversas": []
        }
        save_user(data.username, novo)
        return {"reply": "Conta criada!", "token": token}

    user = load_user(data.username)
    if user["password"] != data.password:
        raise HTTPException(401, "Senha incorreta.")

    token = secrets.token_hex(32)
    user["token"] = token
    save_user(data.username, user)

    return {"reply": f"Bem-vindo, {data.username}!", "token": token}

@app.post("/api/validar_sessao")
def validar_sessao(data: TokenData):
    verificar_sessao(data.username, data.token)
    return {"ok": True}

# ======================================================
# CONVERSAS
# ======================================================

@app.post("/api/conversas")
def listar_conversas(data: TokenData):
    user = verificar_sessao(data.username, data.token)

    lista = []
    for c in user["conversas"]:
        preview = c["mensagens"][-1]["texto"] if c["mensagens"] else ""
        lista.append({
            "id": c["id"],
            "titulo": c["titulo"],
            "preview": preview
        })
    return lista

@app.post("/api/conversa")
def carregar_conversa(data: ChatRequest):
    user = verificar_sessao(data.username, data.token)

    for c in user["conversas"]:
        if c["id"] == data.chat_id:
            return c["mensagens"]

    raise HTTPException(404, "Conversa não encontrada.")

@app.post("/api/nova_conversa")
def nova_conversa(data: NovaConversaData):
    user = verificar_sessao(data.username, data.token)

    cid = "c_" + str(int(datetime.datetime.now().timestamp()))
    nova = {
        "id": cid,
        "titulo": "Nova conversa",
        "mensagens": [],
        "ultimo_acesso": datetime.datetime.now().isoformat()
    }

    user["conversas"].insert(0, nova)
    save_user(data.username, user)

    return {"chat_id": cid}

@app.post("/api/renomear_conversa")
def renomear_conversa(data: RenomearData):
    user = verificar_sessao(data.username, data.token)

    for c in user["conversas"]:
        if c["id"] == data.chat_id:
            c["titulo"] = data.novo_nome
            save_user(data.username, user)
            return {"ok": True}

    raise HTTPException(404, "Não encontrado.")

# ======================================================
# CHAT
# ======================================================

@app.post("/api/chat")
def chat(data: ChatRequest):
    user = verificar_sessao(data.username, data.token)

    conversa = None
    for c in user["conversas"]:
        if c["id"] == data.chat_id:
            conversa = c
            break
    if not conversa:
        raise HTTPException(404, "Conversa não encontrada.")

    conversa["mensagens"].append({"role": "user", "texto": data.message})

    ult = conversa["mensagens"][-2]["texto"] if len(conversa["mensagens"]) > 1 else ""
    diff = (datetime.datetime.now() - datetime.datetime.fromisoformat(conversa["ultimo_acesso"])).total_seconds()
    contexto = f"Último assunto: {ult}" if diff > 3600 else ""

    resposta = AGENT.chat(data.message, contexto_extra=contexto)

    conversa["mensagens"].append({"role": "ai", "texto": resposta})
    conversa["ultimo_acesso"] = datetime.datetime.now().isoformat()

    if conversa["titulo"] == "Nova conversa" and len(conversa["mensagens"]) == 2:
        conversa["titulo"] = AGENT.gerar_titulo(resposta)

    save_user(data.username, user)
    return {"reply": resposta}

# ======================================================
# IMAGEM  (AJUSTADO: Form(...))
# ======================================================

@app.post("/api/imagem")
async def upload_imagem(
    username: str = Form(...),
    token: str = Form(...),
    chat_id: str = Form(...),
    file: UploadFile = File(...)
):
    user = verificar_sessao(username, token)

    ext = file.filename.split(".")[-1]
    fname = f"img_{secrets.token_hex(8)}.{ext}"
    path = f"uploads/{fname}"

    with open(path, "wb") as f:
        f.write(await file.read())

    conv = None
    for c in user["conversas"]:
        if c["id"] == chat_id:
            conv = c
            break

    if not conv:
        raise HTTPException(404, "Conversa não encontrada.")

    # mensagem do usuário
    conv["mensagens"].append({"role": "user", "texto": f"[imagem:{fname}]"})

    # resposta da IA
    resposta = AGENT.chat_imagem(path)
    conv["mensagens"].append({"role": "ai", "texto": resposta})

    save_user(username, user)

    return {"reply": resposta}

# ======================================================
# FRONT-END
# ======================================================

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    ua = request.headers.get("user-agent", "").lower()
    file_name = "mobile.html" if any(x in ua for x in ["iphone", "android", "mobile"]) else "index.html"

    static_file = os.path.join("static", file_name)

    if os.path.exists(static_file):
        return open(static_file, "r", encoding="utf-8").read()

    raise HTTPException(404, "Página não encontrada.")

# Servir arquivos
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")
