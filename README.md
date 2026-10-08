# Treinamento do Qwen3 8B para classificação horária de sepse

Projeto acadêmico que transforma sinais vitais e exames tabulares em prompts curtos e ajusta o modelo oficial [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) por QLoRA. O objetivo experimental é prever o rótulo horário `SepsisLabel` como `0` ou `1`.

O modelo não produz diagnóstico, recomendação terapêutica ou explicação clínica. O dataset é um espelho incompleto do PhysioNet 2019, sem proveniência e licença claramente documentadas. Dados brutos, dados processados, pesos do Qwen e adapters estão excluídos do Git.

## Por que QLoRA

O Qwen3 8B possui aproximadamente 8,2 bilhões de parâmetros. Os pesos BF16 ocupam cerca de 16 GB antes das ativações e do otimizador. A RTX 5060 Ti de 16 GB usada neste projeto exige carregamento NF4 em 4 bits, LoRA, batch 1 e gradient checkpointing.

## Preparação no Windows

Use Python 3.12. No PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt
python -m pip install -e . --no-deps
```

Confirme a GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

## Tratamento dos dados

```powershell
python -m qwen_sepsis.prepare
```

O pipeline:

1. baixa uma revisão fixa do CSV e valida SHA-256;
2. remove uma linha final incompleta;
3. separa pacientes em treino, validação e teste;
4. cria máscaras de ausência antes da imputação;
5. usa somente a última medida das seis horas anteriores;
6. calcula medianas somente no treino;
7. limita o treino a 40 mil exemplos com proporção de três negativos por positivo;
8. preserva a prevalência natural nas amostras de validação e teste.

O balanceamento é aplicado somente ao treino. As saídas ficam em `data/processed` e não são versionadas.

## Download do modelo

O download completo requer aproximadamente 17 GB livres em disco:

```powershell
python -m qwen_sepsis.download_model
```

Os arquivos ficam em `models/base/Qwen3-8B`. Também é possível omitir o download e deixar o Transformers usar o cache do Hugging Face.

## Teste rápido do treinamento

Execute primeiro dois passos com 32 exemplos:

```powershell
python -m qwen_sepsis.train --model-path models/base/Qwen3-8B --smoke
```

Monitore a VRAM em outro terminal:

```powershell
nvidia-smi -l 2
```

## Treinamento completo

```powershell
python -m qwen_sepsis.train --model-path models/base/Qwen3-8B
```

O adapter é salvo em `models/qwen3-8b-sepsis-lora/final_adapter`. Interromper e reiniciar o treinamento ainda não retoma checkpoints automaticamente.

## Resultados do treinamento

O treinamento completou 1 época com loss de `0.24124`, em `69,740.4645` segundos. A velocidade média foi de `0.475` amostras/s (`0.03` passos/s), com `3.2076e17` FLOPs totais. Os valores completos estão em [`reports/train_metrics.json`](reports/train_metrics.json).

A avaliação final ainda está pendente; portanto, estes resultados não incluem métricas de validação ou teste.

## Avaliação

```powershell
python -m qwen_sepsis.evaluate `
  --base-model models/base/Qwen3-8B `
  --adapter models/qwen3-8b-sepsis-lora/final_adapter
```

O limiar é escolhido pela maior F1 na validação e aplicado uma única vez ao teste. As métricas são salvas em `outputs/evaluation.json`. Esse comando continua sendo a avaliação horária.

## Calibração e avaliação por paciente

O protocolo abaixo não retreina o Qwen. Ele lê os logits dos tokens `0` e `1`, calibra na validação, escolhe um limiar de alta sensibilidade e só então avalia o teste por paciente.

Se os dados foram preparados antes deste protocolo, gere o índice de pacientes. O jsonl de treino continua só com `prompt` e `label`:

```powershell
python -m qwen_sepsis.prepare
```

Depois:

```powershell
python scripts/run_calibration.py
```

A meta inicial é 95% de sensibilidade e fica em `configs/calibration.json`. O calibrador e o threshold são escolhidos somente na validation; o test é aberto depois que ambos estão congelados. Os artefatos ficam em `artifacts/calibration`, `artifacts/evaluation` e `artifacts/decision`.

Para regenerar os relatórios a partir dos scores persistidos, sem carregar o Qwen nem recalibrar:

```powershell
python scripts/finalize_decision.py
```

Para pontuar uma observação clínica atual com o Qwen e a decisão congelada, use Python 3.12, GPU CUDA e o exemplo sintético:

```powershell
python scripts/demo_patient.py --input configs/demo_patient_example.json
```

Os campos omitidos são imputados pelas medianas do treino; dados dinâmicos ausentes são marcados no prompt. A resposta é risco binário de suporte à decisão, não diagnóstico. A avaliação atual não alcança sensibilidade horária de 95%; a avaliação patient-level com a regra `any_alert` ainda tem especificidade zero, portanto este CLI é somente demonstração acadêmica.

O resultado é um sistema experimental de apoio à decisão para previsão precoce de risco de sepse. Não substitui o diagnóstico médico.

## Testes de software

Os testes não baixam o Qwen nem o dataset real:

```powershell
python -m pytest -v
```

Eles verificam divisão por paciente, ausência de informação futura, limite temporal de seis horas, validação estrutural, ausência do alvo no prompt e criação dos arquivos processados.

## Limitações

- O rótulo do desafio começa antes do início definido e permanece positivo; a classificação de todas as horas não mede exclusivamente antecipação em seis horas.
- O uso de um LLM para dados tabulares é experimental. O benchmark tabular do repositório `predicao-precoce-sepse` deve permanecer como comparação.
- O treino balanceado altera a distribuição vista pelo modelo. A avaliação horária usa prevalência natural. A calibração e o limiar de alta sensibilidade são definidos só na validação, em `scripts/run_calibration.py`.
- A validação clínica exige outra instituição, revisão das unidades e análise por paciente, hospital e subgrupos.

## Referências

- [Qwen3 8B oficial](https://huggingface.co/Qwen/Qwen3-8B)
- [Quantização com PEFT](https://huggingface.co/docs/peft/developer_guides/quantization)
- [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)

