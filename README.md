# Qwen3-8B para suporte à decisão de risco de sepse

Este repositório implementa uma pipeline de decisão operacional baseada em um modelo já treinado, sem alterar pesos do Qwen3-8B nem o adapter LoRA. A ideia central é: entrada clínica atual do paciente -> prompt do modelo -> score bruto -> calibração -> risco probabilístico -> limiar final -> decisão de suporte à decisão.

A solução é experimental, acadêmica e não substitui diagnóstico médico. Ela tem como objetivo demonstrar a etapa de decisão final sobre risco futuro de sepse usando o modelo preservado, sem novo treinamento e sem re-tuning do adapter.

## Objetivo do projeto

O objetivo desta etapa foi finalizar a parte operativa do sistema:

- manter o modelo base `Qwen/Qwen3-8B` intacto;
- usar o adapter já treinado em `models/qwen3-8b-sepsis-lora/final_adapter`;
- inferir um score para cada observação do paciente;
- calibrar esse score somente na validação;
- escolher um limiar de decisão na validação;
- congelar calibrador e threshold antes de abrir o teste;
- produzir uma demonstração em que o médico informa os dados atuais e recebe um risco estimado.

## Arquitetura real implementada

### 1) Base do modelo e adapter

- Modelo base: `models/base/Qwen3-8B`
- Adapter final: `models/qwen3-8b-sepsis-lora/final_adapter`
- Framework principal: Transformers + PEFT + quantização 4-bit (BitsAndBytes/QLoRA)
- A arquitetura não foi alterada para criar um classificador novo. O modelo continua sendo um LLM causal usado para gerar um score de probabilidade a partir de logits dos tokens `0` e `1`.

### 2) Pipeline de dados

O pré-processamento fica em:

- `src/qwen_sepsis/prepare.py`
- `src/qwen_sepsis/data.py`

A etapa prepara o dataset em três fases:

1. baixa um snapshot fixo do dataset e valida a integridade;
2. separa pacientes em treino, validação e teste;
3. monta prompts com as últimas medidas das seis horas anteriores;
4. aplica imputação por mediana do treino;
5. preserva a prevalência natural nas partições de validação e teste.

O treinamento foi usado apenas para produzir o adapter original; o presente projeto não re-treina o modelo, não ajusta o LoRA e não altera pesos.

### 3) Inferência e pontuação

A inferência está em:

- `src/qwen_sepsis/inference.py`
- `src/qwen_sepsis/explain.py`

A lógica do score é:

1. o prompt contendo sinais vitais e exames do paciente é enviado ao Qwen;
2. o modelo produz logits para os tokens 0 e 1;
3. a probabilidade bruta é inferida via softmax sobre os logits desses tokens;
4. o score bruto é transformado em probabilidade calibrada.

### 4) Calibração e limiar de decisão

A etapa de decisão final está em:

- `src/qwen_sepsis/calibration.py`
- `src/qwen_sepsis/patient_evaluation.py`
- `src/qwen_sepsis/pipeline.py`

O fluxo é:

- `score bruto` -> `calibrador` -> `risk score`
- `risk score` -> `threshold` -> decisão

A regra de seleção foi:

- ajustar o calibrador apenas na validação;
- comparar `raw`, `platt` e `isotonic` por Brier, ECE e log loss;
- escolher o melhor calibrador na validação;
- escolher o limiar também na validação;
- congelar ambos antes de avaliar o teste.

O calibrador final selecionado foi `isotonic`, conforme o melhor desempenho em Brier/ECE/log loss na validação. O limiar escolhido foi `0.01`.

### 5) Regra de decisão operacional

A decisão atual implementada no projeto é uma decisão binária de suporte à decisão:

- `risk_score >= threshold` => risco elevado
- `risk_score < threshold` => baixo risco

Em termos de arquitetura, o sistema é concebido como uma pipeline de risco futuro, mas a implementação executável do demo atual expõe apenas dois estados, não uma triagem clínica formal em três faixas validadas.

### 6) Avaliação por paciente

A avaliação por paciente e a agregação das alertas horários são feitas em:

- `src/qwen_sepsis/patient_evaluation.py`

A regra usada para paciente foi `any_alert`, isto é: se qualquer hora do paciente excedeu o limiar durante o intervalo relevante, o paciente é marcado como positivo para a regra de avaliação por paciente.

Essa agregação é importante para contextualizar o problema, mas também revela uma limitação clínica e estatística importante: a estratégia de paciente otimiza detecção de alertas, mas não garante especificidade operacional aceitável.

## Estado real do desempenho

Os artefatos finais foram gravados em `artifacts/decision` e refletem o comportamento congelado do sistema.

### Seleção do calibrador e limiar

- calibrador escolhido: `isotonic`
- threshold final: `0.01`
- treino e ajuste: nenhum, apenas na validação
- teste usado para ajuste: `false`

### Métrica de validação mais relevante

Na validação, para o limiar final `0.01`:

- sensibilidade: `0.9424`
- especificidade: `0.3189`

Esse resultado representa o melhor compromisso encontrado, mas não atende ao alvo de 95% de sensibilidade com especificidade suficiente. A mensagem do artefato indica explicitamente que nenhum threshold atingiu o requisito mínimo de sensibilidade com uma regra de decisão útil.

### Métrica por paciente

A avaliação por paciente usando `any_alert` foi reportada como:

- sensibilidade: `1.0`
- especificidade: `0.0`

Isso mostra que a regra de paciente detecta todos os pacientes de risco, mas também gera um número muito grande de falsos positivos. Esse é um ponto crítico para interpretação.

## Demonstração operacional

A demonstração de um paciente atual usa:

```powershell
python scripts/demo_patient.py --input configs/demo_patient_example.json
```

Esse comando:

- lê os dados do paciente;
- imputa valores ausentes usando medianas do treino;
- monta o prompt;
- carrega o modelo base + adapter final;
- calcula o score bruto;
- aplica o calibrador frozen;
- compara com o threshold frozen;
- imprime o risco estimado e a mensagem de apoio à decisão.

A saída é um diagnóstico de apoio à decisão e não um diagnóstico médico. A mensagem de disclaimer está incluída na resposta.

## Limitações do trabalho

### 1) Não é um sistema clínico validado

Este projeto é um protótipo acadêmico. Não foi validado em ambiente clínico, não foi testado em outra instituição e não deve ser usado para tomada de decisão médica real.

### 2) Não há garantia de sensibilidade e especificidade operacionais

O limiar final não alcançou a meta de 95% de sensibilidade com especificidade aceitável. O melhor compromisso ficou em `0.9424` de sensibilidade e `0.3189` de especificidade na validação.

### 3) A regra de paciente tem desempenho muito frágil

A regra `any_alert` produz sensibilidade total, mas especificidade zero, o que indica um alto volume de falsos positivos. Em termos práticos, isso significa que o sistema é sensível a sinais de risco, mas também dispara alertas excessivos.

### 4) O problema de sepse e o rótulo são intrinsecamente ambíguos

O rótulo de sepse do desafio começa antes do início da janela clínica usada no problema e permanece positivo durante um intervalo. Isso torna a tarefa mais próxima de uma previsão temporal ambígua do que de uma detecção estritamente binária em um instante fixo.

### 5) O uso de LLM para dados tabulares é experimental

A abordagem usa um modelo generativo em um problema tabular; isso é válido como experimento e como prova de conceito, mas não equivale a uma solução madura de risco clínico. O benchmark tabular específico do problema continua sendo uma referência mais apropriada para comparação.

### 6) O conjunto de dados e o contexto de treinamento não garantem generalização

A base de dados é um espelho do dataset do PhysioNet Challenge 2019 e o projeto usa uma divisão por paciente e prompt específico com medianas do treino. Isso não elimina vieses de hospital, população, qualidade de coleta e convenções de anotação.

### 7) O projeto não produz um sistema de três níveis de risco validado

A arquitetura conceitual de `baixo / intermediário / alto risco` é útil para discussão e apresentação, mas a implementação realmente executável do código final ainda usa uma decisão binária congelada. O regime em três níveis ainda não foi validado com critérios clínicos e estatísticos suficientes.

## Testes e validação de software

Os testes automatizados do repositório são executados com:

```powershell
python -m pytest -v
```

Eles cobrem a divisão por paciente, ausência de informação futura, janela temporal de seis horas, integridade dos dados e geração dos artefatos processados. Não há testes para validade clínica.

## Estrutura principal do repositório

- `configs/` — configurações do calibrador e do exemplo de demonstração
- `data/processed/` — dados já preparados
- `models/base/Qwen3-8B/` — base model
- `models/qwen3-8b-sepsis-lora/final_adapter/` — adapter salvo
- `src/qwen_sepsis/` — código principal do projeto
- `scripts/` — scripts de calibração, decisão e demonstração
- `artifacts/` — artefatos finais, relatórios e métricas
- `reports/` — métricas de treinamento

## Referências

- [Qwen3 8B oficial](https://huggingface.co/Qwen/Qwen3-8B)
- [PEFT / quantização](https://huggingface.co/docs/peft/developer_guides/quantization)
- [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)

## Conclusão

Este repositório finaliza a etapa de decisão de risco de sepse sobre o modelo já treinado, sem treinar de novo nem trocar pesos. O trabalho demonstra um fluxo operacional completo de apoio à decisão, com calibrador e limiar congelados e com uma CLI de teste para observação atual do paciente.

O que está concluído é a cadeia de decisão e a documentação do comportamento real do sistema. O que ainda falta é validação clínica e melhoria estatística do limite operacional para tornar essa solução utilizável em ambiente assistencial real.

