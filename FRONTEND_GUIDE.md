# Guia de frontend para a pipeline de risco de sepse

Este documento explica como transformar a solução atual em uma interface de demonstração operável para médicos e pesquisadores. A ideia não é alterar o modelo, o adapter ou o processo de treinamento. A proposta é reutilizar a pipeline atual de risco e expô-la em um frontend simples, seguro e visual.

## Objetivo

Criar uma interface em que o usuário informe os dados atuais do paciente, o sistema:

1. valida o payload;
2. imputa valores ausentes;
3. monta o prompt;
4. executa a inferência com o Qwen3-8B e o adapter final;
5. aplica o calibrador frozen;
6. compara com o threshold frozen;
7. exibe o risco estimado e a recomendação de suporte à decisão.

A interface deve funcionar como ferramenta de demonstração e validação operacional, não como solução clínica definitiva.

---

## Arquitetura recomendada

### Opção 1: solução rápida e mais simples

Use Streamlit para protótipo funcional em Python, porque o projeto já é Python e a lógica de inferência e imputação está pronta.

Vantagens:

- baixa curva de aprendizado;
- rápida implementação;
- fácil testar a SDK do modelo e a lógica atual;
- ideal para demonstração acadêmica.

Desvantagens:

- menos elegante para produto;
- menos flexível para autenticação e robustez;
- menos apropriado para implantação real em produção.

### Opção 2: solução mais profissional

Use React + TypeScript no frontend e FastAPI no backend.

Vantagens:

- arquitetura mais limpa para produto;
- melhor escalabilidade;
- fácil integração com autenticação, logs, API e UX mais sofisticada;
- apropriada para futuras evoluções.

Desvantagens:

- exige mais desenvolvimento;
- exige organização de backend, rede e deploy;
- maior tempo para chegar ao protótipo funcional.

### Recomendação

Para este projeto, a melhor sequência é:

1. primeiro construir um frontend em Streamlit para validar a experiência;
2. depois migrar para FastAPI + React quando a lógica estiver estável e a solução se tornar demonstrável para usuário final.

---

## Fluxo de dados do frontend

A arquitetura do frontend deve refletir exatamente a pipeline atual implementada no projeto:

```text
Frontend
  -> formulário do paciente
  -> validação do payload
  -> mostrar campos ausentes
  -> chamar backend / API
  -> backend monta prompt
  -> Qwen3-8B + adapter final gera score bruto
  -> calibrador frozen
  -> threshold frozen
  -> retornar risco + mensagem + disclaimer
``` 

## Estrutura sugerida

### Backend (Python)

Crie um serviço Python com duas responsabilidades:

- receber os dados do paciente em JSON;
- executar a lógica já pronta em `scripts/demo_patient.py` em formato de API.

Funções a serem reaproveitadas:

- `prepare_observation()`
- `format_prompt()`
- `load_sepsis_model()`
- `score_prompts()`
- `load_calibrator()`
- `calibrator.predict()`

### Frontend

O frontend deve conter:

- formulário de entrada dos principais sinais vitais e exames;
- campos opcionais para valores ausentes;
- botão “calcular risco”;
- indicação visual da probabilidade calibrada;
- status: baixo risco, risco intermediário ou elevado;
- disclaimer de uso experimental;
- seção de explicação do modelo e das limitações.

---

## Proposta de telas

### 1) Tela inicial

Elementos:

- título do sistema;
- breve explicação: “Suporte à decisão para risco de sepse”;
- botão para iniciar avaliação;
- disclaimer legal e acadêmico.

### 2) Tela de entrada de dados

Campos principais:

- idade;
- pressão arterial sistólica;
- pressão arterial diastólica;
- frequência cardíaca;
- frequência respiratória;
- temperatura;
- SpO2;
- glicose;
- hematócrito / hemoglobina;
- leucócitos;
- creatinina;
- lactato (se disponível);
- outros exames relevantes do dataset.

Cada campo pode ter:

- input numérico;
- opção de “não informado”;
- alerta caso o valor esteja ausente e seja preenchido pela mediana do treino.

### 3) Tela de resultado

Mostrar:

- risco bruto;
- risco calibrado;
- threshold usado;
- status do risco (baixo / elevado);
- mensagem de apoio à decisão;
- lista de campos imputados;
- nota de que o resultado não substitui avaliação clínica.

Exemplo de saída:

```json
{
  "patient_id": "demo-001",
  "raw_score": 0.4378,
  "risk_score": 0.0415,
  "threshold": 0.01,
  "risk_level": "RISCO ELEVADO",
  "calibrator": "isotonic",
  "imputed_features": ["lactate", "spo2"],
  "disclaimer": "Resultado experimental..."
}
```

---

## Backend API proposta

### Endpoint recomendado

`POST /api/patient-risk`

### Payload

```json
{
  "patient_id": "P-001",
  "age": 68,
  "heart_rate": 118,
  "respiratory_rate": 28,
  "temperature": 38.8,
  "systolic_bp": 88,
  "diastolic_bp": 58,
  "spo2": 92,
  "glucose": 176,
  "lactate": null,
  "wbc": 15.4
}
```

### Resposta

```json
{
  "patient_id": "P-001",
  "raw_score": 0.41,
  "risk_score": 0.047,
  "threshold": 0.01,
  "risk_level": "RISCO ELEVADO",
  "message": "Risco elevado de desenvolvimento de sepse neste momento.",
  "imputed_features": ["lactate"],
  "calibrator": "isotonic",
  "test_used_for_fitting": false,
  "disclaimer": "Resultado experimental e não diagnóstico médico."
}
```

---

## Implementação prática com Streamlit

A solução mais rápida é um app em Streamlit. Estrutura sugerida:

```text
app/
  app.py
  utils.py
  config.py
```

### `app.py`

- monta a página;
- cria os controles dos campos;
- faz a chamada para a lógica de inferência;
- mostra o resultado em cards e gráficos.

### `utils.py`

- valida entradas;
- converte `null` em valores ausentes;
- chama a função de score e retorna o JSON.

### Fluxo interno

```python
payload = form_to_dict()
observation, missing = prepare_observation(payload, median_path)
prompt = format_prompt(observation)
records, _ = score_prompts(model, tokenizer, [prompt], batch_size=1)
raw_score = float(records[0]["raw_probability"])
calibrated = calibrator.predict([raw_score])[0]
status = "RISCO ELEVADO" if calibrated >= threshold else "BAIXO RISCO"
```

Essa abordagem reaproveita quase tudo do script atual e reduz muito o tempo de implementação.

---

## Implementação profissional com FastAPI + React

### Backend FastAPI

Estrutura sugerida:

```text
backend/
  main.py
  schemas.py
  service.py
  model_loader.py
```

### Endpoints

- `GET /health`
- `POST /api/patient-risk`
- `GET /api/config`

### Frontend React

Estrutura sugerida:

```text
frontend/
  src/
    components/
    pages/
    api/
    types/
    App.tsx
```

### Componentes

- `PatientForm`
- `RiskSummaryCard`
- `FeatureInputGrid`
- `AlertBanner`
- `DisclaimerBox`

A comunicação pode ser feita por fetch/axios para `POST /api/patient-risk`.

---

## UX e design recomendado

O frontend deve ser claro e cauteloso:

- evitar linguagem que pareça diagnóstico clínico;
- sempre mostrar disclaimer;
- usar cores suaves e sem confusão visual;
- mostrar probabilidade junto ao status e ao threshold;
- incluir explicação de que o sistema é um apoio à decisão e não substitui avaliação do médico.

### Exemplo de UX

- Verde: baixo risco
- Amarelo: atenção / intermediário
- Vermelho: risco elevado

Mas a recomendação prudente é: não criar uma escala sofisticada sem validação clínica. O uso seguro é manter uma interpretação simples: “baixo risco” vs “risco elevado”.

---

## Limitações do frontend

Mesmo com um frontend bem construído, o sistema continua com as mesmas limitações do backend:

- não é clínicamente validado;
- o threshold final ainda tem baixa especificidade;
- a regra por paciente `any_alert` gera muitos falsos positivos;
- o modelo pode produzir risco elevado em muitos casos sem que o paciente realmente evolua para sepse;
- a interface deve reforçar que se trata de suporte à decisão e não diagnóstico.

Isso é importante: o frontend não corrige o problema do modelo; ele apenas torna a decisão acessível e auditável.

---

## Etapas para implementação

### Fase 1 — protótipo funcional

1. reutilizar a lógica de `demo_patient.py`;
2. criar app Streamlit;
3. validar com um paciente de exemplo;
4. exibir risco, threshold e disclaimer.

### Fase 2 — backend API

1. encapsular a lógica em um endpoint REST;
2. separar carregamento do modelo do processamento de cada request;
3. manter o modelo em memória para reduzir latência;
4. adicionar logs e métricas simples.

### Fase 3 — frontend web

1. construir interface em React;
2. incluir validação de campos e feedback visual;
3. mostrar cores e explicações de risco;
4. adicionar histórico de avaliações por paciente.

### Fase 4 — produção e segurança

1. autenticação;
2. controle de acesso;
3. logs de auditoria;
4. documentação da API;
5. testar em ambiente controlado antes de uso clínico.

---

## Recomendação final

O melhor caminho para começar é:

- manter a lógica industrial do modelo atual intacta;
- expor essa lógica em um app de demonstração bem simples;
- usar um frontend que reforça a natureza experimental e o aviso de uso.

Em outras palavras: o frontend deve funcionar como uma interface de apoio à decisão acadêmica, não como uma ferramenta clínica pronta para uso.

---

## Conclusão

A partir do projeto atual, a criação de um frontend é bastante viável e natural. O ponto principal é reutilizar a pipeline que já existe e apenas transformar o fluxo em interface de usuário. O foco deve estar em clareza, segurança, explicação de limitações e reforço do disclaimer clínico.

O objetivo não é convencer o usuário de que o modelo é clínicamente validado. O objetivo é demonstrar de maneira transparente como a decisão de risco é computada e como ela pode ser apresentada de forma operacional.
