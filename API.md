# API de acompanhamento longitudinal

## Escopo e execução

Esta API é um serviço experimental local sobre a pipeline já existente. Ela registra dados fornecidos pelo cliente e executa a inferência congelada quando solicitado; **não recebe dados automaticamente de um hospital e não é monitoramento em tempo real**. Não é validada para uso clínico e não deve receber dados de pacientes reais.

Instale as dependências e o pacote a partir da raiz:

```powershell
python -m pip install -r requirements-gpu.txt -r requirements-api.txt -r requirements-test.txt
python -m pip install -e . --no-deps
python -m uvicorn qwen_sepsis.service_api:app --app-dir src --host 127.0.0.1 --port 8000
```

O serviço usa `data/service.sqlite3` por padrão. O caminho do banco e os diretórios dos artefatos podem ser configurados antes da inicialização:

```powershell
$env:SEPSIS_DB_PATH = "data/service.sqlite3"
$env:SEPSIS_ARTIFACT_DIR = "artifacts"
$env:SEPSIS_DATA_DIR = "data/processed"
$env:SEPSIS_BASE_MODEL = "models/base/Qwen3-8B"
$env:SEPSIS_ADAPTER = "models/qwen3-8b-sepsis-lora/final_adapter"
```

O serviço cria o SQLite e tenta carregar modelo, calibrador e medianas uma vez durante o startup. Uma falha de carregamento deixa a API e o banco disponíveis, mas a prontidão do modelo é `false`; endpoints de inferência respondem `503 MODEL_NOT_READY`. O modelo não é recarregado por requisição. A documentação OpenAPI fica em `GET /docs`.

## Modelo de dados e regras

- `patient_id`, `episode_id`, `observation_id` e `assessment_id` são UUIDs internos. `external_id` é opcional e único; não o use para enviar identificadores pessoais sem controles apropriados.
- A criação de paciente cria atomicamente seu primeiro episódio. Reinternações são representadas por novos episódios.
- Uma observação referencia exatamente um paciente e um episódio. A hora deve ser ISO 8601 com fuso; o banco a normaliza para UTC.
- Os valores recebidos são armazenados como enviados, incluindo `null`; `missing_fields` identifica features omitidas ou nulas. Unidades são opcionais, armazenadas sem conversão e **não entram no prompt do modelo**. A unidade não é especificada pelo contrato do dataset/código de inferência; o chamador é responsável por fornecer valores na representação esperada pelo modelo.
- A inferência recebe somente as 14 features existentes no pipeline: `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `WBC`, `Lactate`, `Creatinine`, `Platelets`, `Age`, `Gender`, `ICULOS`. Campos desconhecidos, tipos incorretos, valores infinitos e datas sem fuso são rejeitados.
- Features ausentes são imputadas pelas medianas do treino. O prompt inclui flags de ausência somente para as features dinâmicas existentes no pipeline (`DYNAMIC`); as demais continuam a ser imputadas sem flag. A resposta indica as imputações dinâmicas como `imputed_features`.
- Uma avaliação só é executada pelo endpoint de assessments ou pelo endpoint pontual `/api/patient-risk`; registrar uma observação **não** executa inferência.
- Cada avaliação guarda score bruto, score calibrado, limiar, decisão binária, método de calibração, versão do modelo, `config_version`, horário, imputações e vínculo à observação. A configuração é fingerprint SHA-256 da configuração congelada, calibrador, medianas e fingerprint do modelo/adaptador.
- Há no máximo uma avaliação por observação e `config_version`. Requisições repetidas retornam a avaliação já persistida. `assessment_id` também é chave de idempotência. O score calibrado significa “probabilidade calibrada na validação”, não probabilidade clínica validada.
- Listagens usam paginação `limit` (1–200, padrão 50) e `offset` (padrão 0). Observações/avaliações podem ser filtradas por `episode_id`; para gráficos, selecione um episódio e use `observed_at` como eixo temporal.

## Contratos REST

Todos os exemplos assumem `http://127.0.0.1:8000`. O objeto `values` pode conter qualquer subconjunto das features reconhecidas; as omitidas e nulas são ausentes.

### `GET /health`

Indica disponibilidade do processo, separada da prontidão do modelo.

```json
{
  "status": "available",
  "model_ready": true,
  "model_error": null
}
```

### `GET /api/config`

Retorna prontidão, versão e configuração congelada. Se o modelo não carregou, os metadados que dependem dos artefatos ficam `null`; features e semântica continuam documentadas.

```json
{
  "model_ready": true,
  "model_id": "Qwen/Qwen3-8B",
  "model_version": "<sha256>",
  "config_version": "<sha256>",
  "calibration_method": "isotonic",
  "calibrator_locked": true,
  "threshold": 0.01,
  "threshold_locked": true,
  "threshold_target_met": false,
  "fitted_on": "validation",
  "test_used_for_fitting": false,
  "features": ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "WBC", "Lactate", "Creatinine", "Platelets", "Age", "Gender", "ICULOS"],
  "dynamic_features": ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "WBC", "Lactate", "Creatinine", "Platelets"],
  "decision_semantics": "RISK_ELEVATED quando risk_score >= threshold; caso contrário LOW_RISK.",
  "risk_score_meaning": "Probabilidade calibrada na validação. Não é diagnóstico nem chance clínica validada.",
  "disclaimer": "Sistema experimental de apoio à decisão clínica para previsão precoce de risco de sepse. Não substitui o diagnóstico médico."
}
```

### `POST /api/patient-risk`

Inferência pontual **sem persistência**; cada chamada executa novo score. Para histórico/idempotência, registre observação e use assessments.

```json
{"values": {"HR": 108, "O2Sat": 95, "Temp": 38.1, "SBP": 102, "MAP": 70, "DBP": 58, "Resp": 23, "WBC": 13.2, "Lactate": null, "Creatinine": 1.1, "Platelets": 172, "Age": 67, "Gender": 1, "ICULOS": 18}}
```

Resposta `200`:

```json
{
  "raw_score": 0.2,
  "risk_score": 0.02,
  "risk_score_meaning": "Probabilidade calibrada na validação. Não é diagnóstico nem chance clínica validada.",
  "threshold": 0.01,
  "decision": "RISK_ELEVATED",
  "calibration_method": "isotonic",
  "model_version": "<sha256>",
  "config_version": "<sha256>",
  "imputed_features": ["Lactate"],
  "message": "Risco elevado de desenvolvimento de sepse.",
  "disclaimer": "Sistema experimental de apoio à decisão clínica para previsão precoce de risco de sepse. Não substitui o diagnóstico médico. Este resultado não constitui diagnóstico médico."
}
```

### `POST /api/patients`

Cria paciente e episódio inicial na mesma transação. `external_id` pode ser omitido.

```json
{"external_id": "demo-001", "initial_episode_label": "internacao-1", "episode_started_at": "2026-10-18T08:00:00-03:00"}
```

Resposta `201`:

```json
{
  "patient": {"patient_id": "11111111-1111-4111-8111-111111111111", "external_id": "demo-001", "created_at": "2026-10-18T11:00:00+00:00"},
  "episode": {"episode_id": "22222222-2222-4222-8222-222222222222", "patient_id": "11111111-1111-4111-8111-111111111111", "label": "internacao-1", "started_at": "2026-10-18T11:00:00+00:00", "created_at": "2026-10-18T11:00:00+00:00"}
}
```

`409` se `external_id` já existir; `422` para corpo inválido.

### `GET /api/patients?limit=50&offset=0`

Exemplo de requisição: `GET /api/patients?limit=50&offset=0`.

Resposta `200`:

```json
{"items": [{"patient_id": "11111111-1111-4111-8111-111111111111", "external_id": "demo-001", "created_at": "2026-10-18T11:00:00+00:00"}], "total": 1, "limit": 50, "offset": 0}
```

Retorna apenas IDs, sem observações ou avaliações embutidas.

### `GET /api/patients/{patient_id}`

Resposta `200`: dados do paciente e lista de episódios; `404` se não encontrado.

```json
{"patient_id": "11111111-1111-4111-8111-111111111111", "external_id": "demo-001", "created_at": "2026-10-18T11:00:00+00:00", "episodes": [{"episode_id": "22222222-2222-4222-8222-222222222222", "patient_id": "11111111-1111-4111-8111-111111111111", "label": "internacao-1", "started_at": "2026-10-18T11:00:00+00:00", "created_at": "2026-10-18T11:00:00+00:00"}]}
```

### `POST /api/patients/{patient_id}/episodes`

Cria episódio para readmissão; o corpo pode omitir `started_at`, que então usa a hora atual UTC.

```json
{"label": "readmissao-2", "started_at": "2026-11-02T07:30:00-03:00"}
```

Resposta `201`: objeto do episódio; `404` se o paciente não existir.

### `GET /api/patients/{patient_id}/episodes`

Exemplo de requisição: `GET /api/patients/11111111-1111-4111-8111-111111111111/episodes`.

Resposta `200`:

```json
[{"episode_id": "22222222-2222-4222-8222-222222222222", "patient_id": "11111111-1111-4111-8111-111111111111", "label": "internacao-1", "started_at": "2026-10-18T11:00:00+00:00", "created_at": "2026-10-18T11:00:00+00:00"}]
```

`404` se o paciente não existir.

### `POST /api/patients/{patient_id}/observations`

O `observation_id` UUID é gerado pelo cliente como chave de idempotência. Repetir o mesmo conteúdo com o mesmo ID retorna `200`; ID com conteúdo diferente ou timestamp repetido em um episódio retorna `409`. Não executa inferência.

```json
{
  "observation_id": "33333333-3333-4333-8333-333333333333",
  "episode_id": "22222222-2222-4222-8222-222222222222",
  "observed_at": "2026-10-18T10:00:00-03:00",
  "values": {"HR": 108, "O2Sat": 95, "Temp": 38.1, "Lactate": null, "Age": 67},
  "units": {"HR": "bpm", "O2Sat": "%", "Temp": "degC", "Lactate": "mmol/L"}
}
```

Resposta `201` (ou `200` para repetição idempotente):

```json
{
  "observation_id": "33333333-3333-4333-8333-333333333333",
  "patient_id": "11111111-1111-4111-8111-111111111111",
  "episode_id": "22222222-2222-4222-8222-222222222222",
  "observed_at": "2026-10-18T13:00:00+00:00",
  "received_at": "2026-10-18T13:01:00+00:00",
  "values": {"HR": 108, "O2Sat": 95, "Temp": 38.1, "Lactate": null, "Age": 67},
  "units": {"HR": "bpm", "O2Sat": "%", "Temp": "degC", "Lactate": "mmol/L"},
  "missing_fields": ["Creatinine", "DBP", "Gender", "ICULOS", "Lactate", "MAP", "Platelets", "Resp", "SBP", "WBC"]
}
```

`404` se paciente/episódio não corresponderem; `409` para duplicidade ou conflito; `422` para UUID, data/hora, unidade ou feature inválidos.

### `GET /api/patients/{patient_id}/observations?episode_id={uuid}&limit=50&offset=0`

Exemplo de requisição: `GET /api/patients/11111111-1111-4111-8111-111111111111/observations?episode_id=22222222-2222-4222-8222-222222222222&limit=50&offset=0`.

`episode_id` é opcional; omiti-lo retorna observações de todos os episódios do paciente, cada qual ainda identificado. Para séries longitudinais, filtrar explicitamente um episódio evita misturar readmissões.

Resposta `200`:

```json
{"items": [{"observation_id": "33333333-3333-4333-8333-333333333333", "patient_id": "11111111-1111-4111-8111-111111111111", "episode_id": "22222222-2222-4222-8222-222222222222", "observed_at": "2026-10-18T13:00:00+00:00", "received_at": "2026-10-18T13:01:00+00:00", "values": {"HR": 108, "O2Sat": 95, "Temp": 38.1, "Lactate": null, "Age": 67}, "units": {"HR": "bpm", "O2Sat": "%", "Temp": "degC", "Lactate": "mmol/L"}, "missing_fields": ["Creatinine", "DBP", "Gender", "ICULOS", "Lactate", "MAP", "Platelets", "Resp", "SBP", "WBC"]}], "total": 1, "limit": 50, "offset": 0}
```

`404` para paciente ou episódio não pertencente ao paciente.

### `POST /api/patients/{patient_id}/assessments`

`assessment_id` é UUID de idempotência do cliente. A API carrega os valores originais da observação e chama a inferência congelada; o chamador não pode escolher threshold ou calibrador.

```json
{"assessment_id": "44444444-4444-4444-8444-444444444444", "observation_id": "33333333-3333-4333-8333-333333333333"}
```

Resposta `201` (ou `200` para repetição):

```json
{
  "assessment_id": "44444444-4444-4444-8444-444444444444",
  "observation_id": "33333333-3333-4333-8333-333333333333",
  "patient_id": "11111111-1111-4111-8111-111111111111",
  "episode_id": "22222222-2222-4222-8222-222222222222",
  "evaluated_at": "2026-10-18T13:02:00+00:00",
  "raw_score": 0.2,
  "risk_score": 0.02,
  "risk_score_meaning": "Probabilidade calibrada na validação. Não é diagnóstico nem chance clínica validada.",
  "threshold": 0.01,
  "decision": "RISK_ELEVATED",
  "calibration_method": "isotonic",
  "model_version": "<sha256>",
  "config_version": "<sha256>",
  "imputed_features": ["Lactate"],
  "message": "Risco elevado de desenvolvimento de sepse.",
  "disclaimer": "Sistema experimental de apoio à decisão clínica para previsão precoce de risco de sepse. Não substitui o diagnóstico médico. Este resultado não constitui diagnóstico médico."
}
```

`404` se a observação não pertencer ao paciente; `409` se `assessment_id` for reutilizado para outra observação; `503 MODEL_NOT_READY` se os artefatos/modelo não estiverem disponíveis. No máximo uma avaliação é gravada por observação e configuração.

### `GET /api/patients/{patient_id}/assessments?episode_id={uuid}&limit=50&offset=0`

Exemplo de requisição: `GET /api/patients/11111111-1111-4111-8111-111111111111/assessments?episode_id=22222222-2222-4222-8222-222222222222&limit=50&offset=0`.

Resposta `200` contém os objetos de avaliação (mesmo formato da resposta `POST /api/patients/{patient_id}/assessments`) dentro de `items`, mais `total`, `limit` e `offset`. Os assessments são históricos e têm versão exata; avaliações de configuração diferente não devem ser agregadas sem indicação visual. `404` para paciente ou episódio não pertencente ao paciente.

## Erros e códigos HTTP

| Código | Uso |
|---|---|
| `200` | Leitura bem-sucedida ou repetição idempotente |
| `201` | Paciente, episódio, observação ou avaliação recém-criados |
| `404` | Paciente, observação ou episódio não encontrado no escopo solicitado |
| `409` | ID/timestamp duplicado, `external_id` existente ou conflito com registro |
| `422` | Corpo, feature, UUID, data/hora, unidade ou paginação inválidos |
| `503` | Serviço disponível, mas modelo/artefatos não prontos para inferência |

Erros de validação usam o formato padrão FastAPI. Erros de restrição retornam mensagem genérica, sem conteúdo clínico.

## Persistência e operação segura

SQLite oferece persistência local ACID e constraints/FKs neste protótipo. O banco usa WAL e espera até 30 segundos por bloqueios. É apropriado para desenvolvimento em uma máquina e carga modesta; não é uma solução multi-host de alta concorrência. Backups, criptografia em repouso, migrações de esquema e política de retenção ainda precisam ser definidos. Para operação multi-instância, migrar para PostgreSQL exige rever migrations, concorrência, backups e credenciais; os dados clínicos não devem ser copiados para ambientes de teste.

A API **não implementa autenticação, autorização, auditoria de acesso, criptografia de transporte/armazenamento, segregação por organização, consentimento, retenção ou remoção**. Antes de qualquer uso com pacientes reais, são obrigatórios controles de identidade e acesso por papel, TLS, gestão segura de chaves/segredos, auditoria sem registrar payloads, proteção/backup do banco, retenção compatível com a regulação, avaliação de privacidade e validação clínica/regulatória. O servidor local vincula a `127.0.0.1` por padrão documentado; não o exponha numa rede sem esses controles. Logs da aplicação não incluem dados clínicos; configurar/desabilitar access logs do servidor se UUIDs das rotas também forem considerados sensíveis.

## Estado e validação do modelo

A configuração atualmente congelada usa calibrador isotônico e threshold `0.01`, ajustados na validação; a meta de sensibilidade horária de 95% **não foi atingida** (sensibilidade `0.9424`, especificidade `0.3189`). Na validação, a seleção do calibrador foi avaliada no mesmo conjunto usado para ajustar os calibradores, portanto Brier/ECE são in-sample. No relatório de pacientes do teste, a regra `any_alert` produziu sensibilidade `1.0`, especificidade `0.0` e `466` falsos positivos entre `466` pacientes sem sepse. Esse desempenho é incompatível com uso operacional seguro. Um relatório de qualidade antigo também lista threshold raw `0.0`; ele não substitui o threshold congelado de inferência (`0.01`).

Uma avaliação retrospectiva longitudinal adequada ainda precisa:

1. Fixar previamente coorte, instante de previsão, horizonte(s) temporais, regra de onset e tratamento de observações ausentes/gaps.
2. Separar por paciente antes de qualquer escolha, mantendo a ordem temporal; nenhuma observação futura pode entrar em features, imputação, calibração ou seleção de threshold.
3. Selecionar calibrador, threshold e regra temporal somente na validação; congelar tudo antes de abrir o teste, sem reotimização no teste.
4. No conjunto de teste por paciente, simular previsões causais hora a hora, medir sensibilidade/especificidade/PPV/NPV, AUROC/AUPRC, calibração, falsos alertas por paciente e distribuição do lead time antes do primeiro onset; reportar incerteza e subgrupos.
5. Tratar as métricas existentes como evidência de risco: a inferência do modelo é por observação horária e a regra `any_alert` não equivale, por si só, a previsão longitudinal validada.

Nenhuma etapa acima é executada automaticamente pela API. O threshold não pode ser alterado via request e o conjunto de teste não é usado para ajustar configuração.

## Testes

```powershell
python -m pytest tests/test_service_api.py tests/test_calibration.py -v
```

Para uma checagem manual, compare `GET /api/config`, crie paciente/observação e solicite uma avaliação usando os exemplos acima. A execução real exige GPU/CUDA compatível com a quantização BitsAndBytes usada pelo pipeline, pesos locais, adapter, calibrador e `train_medians.csv`. A API não substitui testes clínicos nem a avaliação retrospectiva temporal descrita acima.
