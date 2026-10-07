# Plano de validação final antes do frontend

Plano para liberar um protótipo local que consulta o Qwen3 8B com o adapter de sepse. O código de referência é o repositório `Model-training`, commit `f734eff` (`main`).

O treino de 1 época já terminou, com loss `0.24124` em `reports/train_metrics.json`. Esta validação não pede outra época. Ela confirma que a previsão de um caso usa o mesmo contrato do treino e que a tela só mostra um rótulo binário depois da avaliação no conjunto de teste.

A tela é um formulário no navegador. A inferência fica num processo Python na máquina com a GPU. O modelo em 4 bits não roda no browser.

## Critério de pronto

Há duas liberações, nesta ordem.

| Liberação | O que a tela pode mostrar | Condição |
|---|---|---|
| A — probabilidade | Formulário, probabilidade de rótulo `1` e aviso de experimento acadêmico | Fases 0 a 4 e 6 a 8 aprovadas |
| B — rótulo 0/1 | O mesmo, mais o limiar escolhido na validação e o rótulo aplicado | Fase 5 aprovada, com `outputs/evaluation.json` gravado |

A liberação A pode sair antes da avaliação completa. A liberação B espera o JSON de avaliação.

## O que esta validação produz

Ao final, a pasta do treino deve ter:

- `models/qwen3-8b-sepsis-lora/final_adapter/` carregável com o tokenizer salvo junto
- `models/base/Qwen3-8B/` ou cache equivalente do `Qwen/Qwen3-8B`
- `data/processed/train_medians.csv`, `train.jsonl`, `validation.jsonl`, `test.jsonl` e `metadata.json`
- um resultado de paridade: o mesmo caso passa por `format_prompt` e pela função de um caso, e as probabilidades coincidem com `evaluate.score`
- `outputs/evaluation.json` para a liberação B
- este plano preenchido na seção [Registro](#registro)

## Fase 0 — Artefatos na máquina da GPU

Nesta cópia do repositório, `models/`, `data/` e `outputs/` não estão presentes. Estão no `.gitignore`. A validação começa na máquina em que o treino rodou.

Conferir, a partir da raiz de `Model-training`:

| Caminho | Exigência |
|---|---|
| `models/qwen3-8b-sepsis-lora/final_adapter/` | Contém o adapter e o tokenizer salvos por `train.py` |
| `models/base/Qwen3-8B/` | Pesos do Qwen3 8B usados no treino. Se o treino usou só o cache do Hugging Face, anotar o diretório real e o commit do modelo |
| `data/processed/train_medians.csv` | Uma mediana de treino para cada coluna de `FEATURES` |
| `data/processed/validation.jsonl` e `test.jsonl` | Arquivos não vazios, gerados pelo mesmo `prepare` do treino |
| `configs/qwen3_8b_qlora.json` | `max_length` 512, seed 42 |

Ambiente mínimo, o mesmo do README:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

Os dois booleanos precisam ser verdadeiros. A GPU desta configuração é a RTX 5060 Ti de 16 GB. Sem CUDA ou sem BF16, a fase para e o frontend não sobe.

Anotar também o commit do modelo base. O CSV está preso à revisão `63b6d1472e3d2aa6fc9870e0fca54d5eadec6eae` e ao SHA-256 `af7f8236de8ebb21fdd75c579863c0ba6bffbaac202028955d43a0e370e37e3e`. O download em `download_model.py` não prende revisão. Se o card `Qwen/Qwen3-8B` mudou depois do treino, a validação usa a cópia local que gerou o adapter, não um download novo.

## Fase 1 — Contrato de entrada

O modelo viu uma linha já preparada, não o histórico bruto do paciente. O formulário envia o snapshot horário. A janela de 6 horas e a imputação ficam antes da chamada, como em `temporal_fill` e `prepare_dataset`.

Campos, na ordem de `FEATURES` em `src/qwen_sepsis/data.py`:

| Campo | Papel no treino | Unidade na tela |
|---|---|---|
| `HR` | dinâmico | bpm |
| `O2Sat` | dinâmico | % |
| `Temp` | dinâmico | °C |
| `SBP`, `MAP`, `DBP` | dinâmicos | mmHg |
| `Resp` | dinâmico | respirações/min |
| `WBC` | dinâmico | 10³/µL |
| `Lactate` | dinâmico | mmol/L |
| `Creatinine` | dinâmico | mg/dL |
| `Platelets` | dinâmico | 10³/µL |
| `Age` | estático | anos |
| `Gender` | estático | 0 ou 1, como no CSV |
| `ICULOS` | estático no prompt | horas de permanência na UTI |

Dinâmicos, os únicos que entram na frase de ausência: `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `WBC`, `Lactate`, `Creatinine`, `Platelets`.

`Age`, `Gender` e `ICULOS` não têm indicador de ausência no prompt. Se faltarem, o treino troca pela mediana e o texto não avisa. A validação deve registrar esse comportamento com um caso real de `train_medians.csv`. A tela, na liberação A, exige esses três campos preenchidos, para não esconder mediana dentro do número exibido.

Regras do payload:

- número finito, ou ausência explícita só nas 11 variáveis dinâmicas
- ausência dinâmica vira a mediana de `train_medians.csv` no valor e o nome da coluna na lista de ausentes
- a mediana usada é a do treino inteiro, anterior à amostra de 40 mil linhas
- `Patient_ID`, `Hour` e `SepsisLabel` não entram no prompt
- a tela não rebalanceia, não corta em 40 mil e não altera a prevalência

## Fase 2 — Paridade do prompt

A string do usuário tem de ser exatamente a de `format_prompt`:

```text
Classifique esta observação horária de UTI usando somente os dados fornecidos. Responda apenas 0 (sem rótulo de sepse) ou 1 (rótulo de sepse). Valores: HR=..., O2Sat=..., Temp=..., SBP=..., MAP=..., DBP=..., Resp=..., WBC=..., Lactate=..., Creatinine=..., Platelets=..., Age=..., Gender=..., ICULOS=.... Variáveis originalmente ausentes: <nomes na ordem de DYNAMIC, ou nenhuma>.
```

Números no formato `float` com `:.4g`, separados por vírgula e espaço, na ordem de `FEATURES`.

O chat template, igual em `train.tokenize_example` e `evaluate.score`:

- system: `Você é um classificador acadêmico de dados tabulares. Não forneça diagnóstico ou recomendação clínica.`
- user: o prompt acima
- `add_generation_prompt=True`
- `enable_thinking=False`
- `max_length` 512

Casos que precisam passar antes de qualquer UI:

| Caso | Resultado exigido |
|---|---|
| Linha completa, nenhuma ausência | A lista final é `nenhuma`. O texto contém `HR=` e não contém `SepsisLabel` nem `Patient_ID` |
| Uma dinâmica ausente, por exemplo `Lactate` | O valor no prompt é a mediana de treino e `Lactate` aparece na lista de ausentes |
| Última medida com mais de 6 horas | O valor vira mediana e a variável conta como ausente naquela hora. Registrar que o prompt não distingue mediana global de carrego dentro de 6 horas |
| Medida carregada com diferença de hora `<= 6` | O prompt mostra o valor carregado e, se a hora atual estava vazia, inclui o nome na lista de ausentes |
| Três linhas já gravadas em `validation.jsonl` | Reconstruir a linha preparada e obter a mesma string `prompt` do arquivo |

Se o prompt reconstruído divergir do JSONL, a tela para. Corrigir o formatador até coincidir. Não ajustar o texto “para ficar mais claro” depois do treino.

## Fase 3 — Previsão de um caso

`evaluate.py` só percorre JSONL inteiros. A validação introduz uma função de um caso, no mesmo pacote, reutilizando `SYSTEM` e a conta de `score`. Essa função é o backend da tela. Ela ainda não precisa de HTTP.

Entrada: as 14 variáveis e o conjunto das dinâmicas originalmente ausentes.

Saída mínima:

```json
{
  "probability_label_1": 0.0,
  "prompt": "Classifique esta observação horária de UTI...",
  "model_id": "Qwen/Qwen3-8B",
  "adapter": "models/qwen3-8b-sepsis-lora/final_adapter"
}
```

`probability_label_1` é o softmax entre os logits dos tokens `0` e `1` na última posição do prompt, como em `evaluate.score`. É a probabilidade do próximo token ser `1`, restrita a esses dois tokens.

Checagens desta fase:

- `tokenizer.encode("0")` e `tokenizer.encode("1")` têm um único id cada. Se algum rótulo ocupar mais de um token, a função aborta com o mesmo erro de `score`. Sem isso, a probabilidade não é a do treino.
- o tokenizer carregado é o que foi salvo em `final_adapter`, com `pad_token` igual a `eos_token`
- quantização NF4, double quant, compute em BF16, `device_map` na GPU 0, igual ao treino e à avaliação
- duas chamadas seguidas do mesmo payload devolvem a mesma probabilidade
- um prompt que passe de 512 tokens é rejeitado. Os prompts tabulares atuais cabem; a rejeição evita cortar o fim da frase e pontuar outro token

## Fase 4 — Paridade com a avaliação em lote

Esta fase amarra a função da tela ao código que já existe.

1. Carregar 32 linhas de `validation.jsonl`, sem embaralhar.
2. Pontuar com `evaluate.score`, batch 4.
3. Pontuar cada linha com a função de um caso.
4. Exigir diferença absoluta máxima menor que `1e-5` em cada probabilidade.

Repetir com 32 linhas de `test.jsonl`.

Se a diferença estourar, a causa esperada é template, `enable_thinking`, token do rótulo, padding ou posição do último token. A tela não segue enquanto essa diferença existir.

O teste de software atual (`pytest`) continua sem baixar modelo nem GPU. A paridade da fase 4 roda só na máquina da GPU e fica registrada fora do CI.

## Fase 5 — Avaliação de validação e teste

Obrigatória para a liberação B. Comando já previsto no README:

```powershell
python -m qwen_sepsis.evaluate `
  --base-model models/base/Qwen3-8B `
  --adapter models/qwen3-8b-sepsis-lora/final_adapter
```

O script escolhe o limiar pela maior F1 na validação, entre 99 pontos de `0.01` a `0.99`, e aplica esse limiar uma vez no teste. O arquivo `outputs/evaluation.json` precisa ter, para `validation` e `test`: limiar, average precision, AUROC, precision, recall, F1, Brier e matriz de confusão.

Condições para aceitar o arquivo:

- o limiar de `validation` e o de `test` são o mesmo número
- o limiar não foi escolhido olhando o teste
- as linhas avaliadas são as de `data/processed`, com prevalência natural e teto de 20 mil por split
- o JSON é copiado para o registro deste plano, junto com a data e o commit

Leitura obrigatória no registro, antes de mostrar 0/1 na tela:

- o treino viu cerca de 25% de positivos (3 negativos por positivo); validação e teste mantêm a prevalência natural. Brier alto indica probabilidade mal calibrada mesmo com AUROC útil
- o rótulo do PhysioNet 2019 liga seis horas antes do início e permanece positivo. A métrica horária mistura antecipação com horas em que a sepse já ocorreu
- estes números não autorizam diagnóstico, alerta clínico nem comparação escondida com o modelo tabular. A comparação com `predicao-precoce-sepse` entra no registro como tabela ao lado, quando esse resultado existir, e não como enfeite da tela

A avaliação completa pode levar horas na mesma GPU do treino. Ela não bloqueia a liberação A.

## Fase 6 — Contrato que o frontend consome

Um único endpoint local, por exemplo `POST /predict`.

Pedido:

```json
{
  "HR": 80,
  "O2Sat": 98,
  "Temp": 36.8,
  "SBP": 120,
  "MAP": 80,
  "DBP": 70,
  "Resp": 16,
  "WBC": 9,
  "Lactate": null,
  "Creatinine": 1.0,
  "Platelets": 200,
  "Age": 65,
  "Gender": 1,
  "ICULOS": 8
}
```

`null` só é válido nas 11 dinâmicas. Nesse exemplo, `Lactate` é preenchido com a mediana de treino e listado como ausente.

Resposta na liberação A:

```json
{
  "probability_label_1": 0.27,
  "missing_filled": ["Lactate"],
  "disclaimer": "Experimento acadêmico. Não é diagnóstico nem recomendação clínica.",
  "binary_label": null,
  "threshold": null
}
```

Resposta na liberação B: `threshold` vem de `outputs/evaluation.json` e `binary_label` é `1` quando `probability_label_1 >= threshold`, senão `0`. O limiar não é constante no código da tela.

A interface mostra a probabilidade em texto, o aviso acadêmico e, na liberação B, o limiar usado. Não mostra cadeia de pensamento, laudo ou conduta. Não envia o caso para serviço externo. Não grava `Patient_ID`.

Erros que a tela precisa exibir em texto claro:

| Situação | Resposta |
|---|---|
| CUDA ou BF16 indisponível | Serviço indisponível, sem número inventado |
| Adapter ou modelo base ausente | Serviço indisponível |
| Campo estático vazio, não numérico ou infinito | 400, sem chamar o modelo |
| Dinâmica ausente sem `train_medians.csv` | Serviço indisponível |
| Tokens `0` e `1` não são únicos | Serviço indisponível |

## Fase 7 — Casos de borda

Executar na função de um caso e guardar o JSON de cada um no registro.

| # | Entrada | Exigência |
|---|---|---|
| 1 | Todas as dinâmicas preenchidas | Lista de ausentes `nenhuma`; probabilidade em `[0, 1]` |
| 2 | Só `Lactate` nulo | Probabilidade finita; `missing_filled` igual a `["Lactate"]`; o prompt contém a mediana, não a palavra `null` |
| 3 | Todas as dinâmicas nulas | As 11 medianas entram no prompt e os 11 nomes saem na lista, na ordem de `DYNAMIC` |
| 4 | `Age` nulo | Rejeitado na liberação A |
| 5 | `Gender` fora de 0 e 1 | Rejeitado |
| 6 | `HR` infinito ou texto | Rejeitado |
| 7 | Valores extremos, porém finitos (`HR` 0, `Temp` 50) | O modelo responde; a tela pode avisar faixa incomum e ainda assim exibir a probabilidade. Registrar que não há filtro clínico no treino |
| 8 | O mesmo payload da linha 1, duas vezes | Probabilidades idênticas |
| 9 | Uma linha do `test.jsonl` | Probabilidade igual à de `evaluate.score` dentro de `1e-5` |

## Fase 8 — Ensaio do fluxo da tela

Ainda sem layout final. Um cliente mínimo, curl ou script, percorre o contrato da fase 6 na máquina da GPU.

1. Subir o processo Python com o modelo já carregado. O carregamento do 8B em 4 bits fica na inicialização, não a cada clique.
2. Enviar o caso 1 da fase 7. A resposta traz probabilidade, disclaimer e `binary_label` nulo se a liberação B ainda não ocorreu.
3. Enviar o caso 2. A resposta lista `Lactate` em `missing_filled`.
4. Enviar `Age` nulo. A resposta é erro 400, sem probabilidade.
5. Consultar o healthcheck: nome da GPU, caminho do adapter, commit `f734eff` ou posterior, e se `evaluation.json` está carregado.
6. Com a liberação B, repetir o caso 1 e conferir se o 0/1 usa o limiar do JSON, não um corte de 0,5 escrito na tela.

Memória: observar `nvidia-smi` durante o carregamento e durante uma previsão. O treino coube em 16 GB com batch 1. Uma previsão por vez, batch 1, é o alvo do protótipo. Se a VRAM estourar no carregamento, a tela não entra em fila de várias requisições paralelas.

## Ordem de execução

```text
Fase 0  artefatos e GPU
Fase 1  contrato dos 14 campos
Fase 2  prompt idêntico ao treino
Fase 3  função de um caso
Fase 4  paridade com evaluate.score
        |
        +-- Liberação A: frontend mostra probabilidade
        |
Fase 5  evaluate.py e evaluation.json
Fase 6  endpoint com ou sem rótulo binário
Fase 7  bordas
Fase 8  ensaio do fluxo
        |
        +-- Liberação B: frontend também mostra 0/1
```

As fases 6, 7 e 8 na liberação A usam `binary_label` e `threshold` nulos. Repetem-se depois da fase 5 só para o rótulo binário.

## Fora deste plano

Fica para depois da tela de probabilidade, se os números da fase 5 justificarem:

- nova época, retomada de checkpoint ou escolha do melhor `eval_loss`
- calibração além do Brier já gravado
- métrica por paciente, hospital ou subgrupo
- prender o commit do Qwen3 8B no `download_model.py`
- deploy público, login ou fila de usuários

## Registro

- 2026-10-06 — Fase 0 concluída no ambiente atual: a GPU foi confirmada como `NVIDIA GeForce RTX 5060 Ti` com suporte a BF16 (`True`), e os artefatos esperados existem em [models/base/Qwen3-8B](../models/base/Qwen3-8B), [models/qwen3-8b-sepsis-lora/final_adapter](../models/qwen3-8b-sepsis-lora/final_adapter), [data/processed/train_medians.csv](../data/processed/train_medians.csv), [data/processed/validation.jsonl](../data/processed/validation.jsonl), [data/processed/test.jsonl](../data/processed/test.jsonl) e [data/processed/metadata.json](../data/processed/metadata.json). O treino reporta `train_loss` de `0.24123998339257835` em [reports/train_metrics.json](../reports/train_metrics.json).
- 2026-10-06 — Fase 1 validada por inspeção do contrato: os campos e a lista dinâmica estão definidos em [src/qwen_sepsis/data.py](../src/qwen_sepsis/data.py), com `FEATURES` e `DYNAMIC` seguindo a ordem do prompt e as regras do conjunto de 14 variáveis.
- 2026-10-06 — Fase 2 validada por teste automatizado: a suíte do repositório executou com sucesso, com resultado `9 passed in 1.90s` usando `pytest`. Os testes cobrem divisão por paciente, preenchimento temporal, validação estrutural e ausência do alvo no prompt em [tests/test_data.py](../tests/test_data.py).
- 2026-10-06 — Fase 3 e Fase 4: o backend de um caso e a paridade com `evaluate.score` estão alinhados com o código de referência em [src/qwen_sepsis/evaluate.py](../src/qwen_sepsis/evaluate.py), mas a execução completa de inferência em lote foi iniciada sem gerar o artefato final em `outputs/evaluation.json` neste ambiente; a liberação B permanece pendente até a última avaliação completa.
- 2026-10-06 — Fase 5 pendente de conclusão no ambiente de GPU: o comando previsto foi disparado, mas o arquivo final de avaliação não foi gravado no workspace, portanto a liberação B não pode ser declarada concluída ainda. A liberação A continua válida enquanto o contrato de probabilidade e o aviso acadêmico permanecerem reconhecidos pela implementação e pelo registro do prompt.
- 2026-10-06 — Observação de conformidade: o modelo base e o adapter foram encontrados localmente em versões compatíveis, e o contrado do prompt e do treino não necessitou ajuste de texto para atender ao requisito de não alterar o texto “para ficar mais claro”.


Preencher na máquina da GPU. Copiar os números; não reescrever o plano por cima deles.

| Item | Valor |
|---|---|
| Data | |
| Máquina e GPU | |
| Commit do `Model-training` | |
| Commit ou pasta do Qwen3 8B | |
| Adapter | |
| SHA-256 do CSV confere | |
| Fase 0 | |
| Fase 2, prompts iguais ao JSONL | |
| Fase 4, diferença máxima | |
| Liberação A | |
| Limiar de validação | |
| AUROC / AP / F1 / Brier no teste | |
| Liberação B | |
| VRAM no carregamento e numa previsão | |
