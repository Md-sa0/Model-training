# Frontend HealthTech experimental

Interface React + TypeScript isolada em `frontend/`, conectada à API longitudinal descrita em `..\API.md`. O visual escuro com acentos verde-água e painel lateral usa `..\image-front.png` como referência, sem copiar métricas ou pacientes ilustrativos.

## Requisitos e execução no Windows

Use Node.js 22.12+ ou 24+ e Python com o ambiente de dependências da API/modelo configurado. Instale o frontend uma vez:

```powershell
cd frontend
npm install
```

Inicie em **dois terminais** a partir da raiz do repositório:

**Terminal 1 — API**

```powershell
python -m pip install -r requirements-gpu.txt -r requirements-api.txt -r requirements-test.txt
python -m pip install -e . --no-deps
python -m uvicorn qwen_sepsis.service_api:app --app-dir src --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend**

```powershell
cd frontend
npm run dev
```

Abra o endereço informado pelo Vite, normalmente `http://localhost:5173`. O `.env.development` define `VITE_API_BASE_URL=http://127.0.0.1:8000`; o proxy do Vite encaminha `/api` e `/health` para a API. Para mudar o destino local, edite essa variável ou copie `.env.example` para `.env.local`. Não coloque credenciais ou segredos em variáveis `VITE_*`: elas são incorporadas ao bundle do navegador.

Em produção o frontend usa `VITE_API_BASE_URL` diretamente. Para hospedar a UI em origem diferente da API, o servidor API precisa permitir CORS ou deve ser configurado um proxy same-origin externo; a API atual não configura CORS.

## Funcionalidades e limites

- Dashboard consulta `/health`, `/api/config` e lista paginada `/api/patients`.
- Cadastros e consulta longitudinal usam os contratos implementados de pacientes, episódios, observações e avaliações.
- Só a ação explícita de avaliação chama inferência. Um `503 MODEL_NOT_READY` é exibido sem simular resultado.
- Gráficos são cálculos locais usando apenas os registros retornados na página atualmente carregada; não são agregações da API nem monitoramento em tempo real.
- A interface não armazena dados clínicos no `localStorage`. A API não oferece autenticação; não use com dados clínicos reais.
- O modelo e seus scores são experimentais e não validados para decisões clínicas. `LOW_RISK` não garante ausência de sepse. Confirme unidades e faixas de entrada com o protocolo do projeto.

## Verificações

```powershell
cd frontend
npm run lint
npm test -- --run
npm run build
```
