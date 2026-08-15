# Spirit Backend

API FastAPI com chat em streaming pela Nvidia, memória persistente em SQLite e pesquisa web via DuckDuckGo.

## Instalação e execução

No diretório deste projeto, crie e ative um ambiente virtual (opcional), instale as dependências e crie o `.env` a partir do exemplo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edite `.env` e defina `NVIDIA_API_KEY`. Em seguida:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Documentação interativa: `http://127.0.0.1:8000/docs`.

## Endpoints

- `GET /status`: estado atual.
- `POST /power`: alterna o estado ativo.
- `POST /chat`: `{ "message": "Olá", "conversation_id": "opcional" }`. A resposta é texto em streaming; guarde o cabeçalho `X-Conversation-ID` para continuar a conversa.
- `POST /research`: `{ "term": "FastAPI", "max_results": 5 }`. Retorna resumo e fontes. A busca usa snippets públicos; eles podem estar desatualizados, então abra as URLs antes de tomar decisões importantes.

Exemplo de pesquisa:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/research -ContentType 'application/json' -Body '{"term":"FastAPI"}'
```

## Observações de segurança

- A chave existe somente no ambiente/arquivo `.env`, nunca no código ou frontend.
- O CORS aceita os endereços locais comuns e `null` para frontend aberto via `file://`. Defina `CORS_ORIGINS` explicitamente antes de publicar.
- O arquivo SQLite (`spirit.db`) é criado localmente. Faça backup ou remova-o se quiser apagar a memória.
