# Spirit

**Spirit** é um projeto de assistente de IA pessoal, composto por um frontend leve em HTML, CSS e JavaScript puro e uma API em FastAPI. O backend conversa com o modelo Nemotron 3.5 Lightning pela API da Nvidia, mantém memória local em SQLite, pesquisa a web e cria resumos de conteúdos do YouTube.

## Estrutura do projeto

```text
Spirit/
├── frontend/                 # Interface web: HTML, CSS e JavaScript
├── backend/                  # API FastAPI e dependências Python
│   ├── main.py               # Aplicação e endpoints da API
│   ├── requirements.txt      # Dependências Python
│   ├── .env                  # Configuração local (não versionar)
│   └── spirit.db             # Memória local SQLite
├── docs/                     # Documentação complementar
└── README.md
```

## Requisitos

- Python 3.10 ou superior
- Uma chave de API da Nvidia com acesso ao modelo configurado

## Instalação

No terminal, entre na pasta do backend e crie um ambiente virtual:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

## Configuração

Crie o arquivo `backend/.env`:

```env
NVIDIA_API_KEY=sua_chave_aqui
```

Configurações opcionais:

```env
NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
CORS_ORIGINS=http://localhost:5500,http://127.0.0.1:5500
SPIRIT_DATABASE=spirit.db
LOG_LEVEL=INFO
```

> **Segurança:** `NVIDIA_API_KEY` é uma chave pessoal e nunca deve ser incluída em commits, issues, capturas de tela ou código do frontend. Mantenha `.env` no `.gitignore`.

## Executando

Com o ambiente virtual ativo, ainda em `backend/`:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

A API ficará disponível em `http://127.0.0.1:8000` e a documentação interativa em `http://127.0.0.1:8000/docs`.

Para servir o frontend localmente:

```powershell
cd frontend
python -m http.server 5500
```

Abra `http://127.0.0.1:5500` no navegador.

## API

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/chat` | Envia uma mensagem para a Spirit. A resposta é transmitida como NDJSON, com eventos `reasoning` e `content`. |
| `POST` | `/research` | Pesquisa a web via DuckDuckGo e resume os resultados com o modelo da Nvidia. |
| `POST` | `/research/youtube` | Localiza vídeos do YouTube, obtém transcrições disponíveis e gera um resumo. Vídeos sem legenda são ignorados. |
| `GET` | `/conversations` | Lista conversas salvas na memória SQLite. |
| `GET` | `/conversations/{conversation_id}/messages` | Retorna mensagens persistidas de uma conversa específica. |
| `GET` | `/status` | Informa se a Spirit está ativa. |
| `POST` | `/power` | Alterna o estado ativo/desativado da Spirit. |

### Exemplos

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/research `
  -ContentType 'application/json' `
  -Body '{"term":"FastAPI","max_results":5}'
```

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/research/youtube `
  -ContentType 'application/json' `
  -Body '{"term":"introdução ao FastAPI","max_results":3}'
```

## Memória local

As mensagens são armazenadas no SQLite definido em `SPIRIT_DATABASE` — por padrão, `spirit.db`. Para continuar uma conversa, envie o mesmo `conversation_id` recebido no cabeçalho `X-Conversation-ID` da primeira chamada ao `/chat`.
