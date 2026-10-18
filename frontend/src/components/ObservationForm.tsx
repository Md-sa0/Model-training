import { zodResolver } from "@hookform/resolvers/zod";
import { useEffect, useMemo } from "react";
import { useForm } from "react-hook-form";
import { useCreateObservation } from "../hooks/useApi";
import { toApiError } from "../api/client";
import { buildObservationPayload, observationFormSchema } from "../lib/schemas";
import { featureKeys } from "../types/api";
import type { Episode, FeatureKey, Observation } from "../types/api";
import type { z } from "zod";
import { Alert, Button, Field, SelectInput, TextInput } from "./ui";

type ObservationFields = z.infer<typeof observationFormSchema>;

export function ObservationForm({
  patientId,
  episodes,
  selectedEpisodeId,
  onSaved,
}: {
  patientId: string;
  episodes: Episode[];
  selectedEpisodeId: string;
  onSaved: (observation: Observation) => void;
}) {
  const initialValues = useMemo(
    () => Object.fromEntries(featureKeys.map((key) => [key, ""])) as Record<FeatureKey, string>,
    [],
  );
  const form = useForm<ObservationFields>({
    resolver: zodResolver(observationFormSchema),
    defaultValues: {
      episode_id: selectedEpisodeId,
      observed_at: "",
      values: initialValues,
      units: { ...initialValues },
    },
  });
  const createObservation = useCreateObservation(patientId);
  useEffect(() => {
    form.setValue("episode_id", selectedEpisodeId);
  }, [form, selectedEpisodeId]);

  async function submit(input: ObservationFields) {
    try {
      const payload = buildObservationPayload(input, crypto.randomUUID());
      const result = await createObservation.mutateAsync(payload);
      onSaved(result);
      form.reset({ ...form.getValues(), observed_at: "", values: initialValues, units: { ...initialValues } });
    } catch {
      // Mutation error is rendered below.
    }
  }

  const apiError = createObservation.error ? toApiError(createObservation.error) : null;
  const episodeError = form.formState.errors.episode_id?.message;

  return (
    <form className="form-stack observation-form" onSubmit={form.handleSubmit(submit)} noValidate>
      {apiError && (
        <Alert tone="error" title={apiError.status === 409 ? "Conflito ao registrar observação" : "Não foi possível salvar"}>
          {apiError.message}
          {apiError.validation.length > 0 && <ul>{apiError.validation.map((item) => <li key={item}>{item}</li>)}</ul>}
        </Alert>
      )}
      <div className="form-grid form-grid-two">
        <Field label="Episódio" htmlFor="observation-episode" error={episodeError}>
          <SelectInput id="observation-episode" {...form.register("episode_id")}>
            <option value="">Selecione um episódio</option>
            {episodes.map((episode) => (
              <option key={episode.episode_id} value={episode.episode_id}>
                {episode.label || episode.episode_id} · {new Date(episode.started_at).toLocaleDateString("pt-BR")}
              </option>
            ))}
          </SelectInput>
        </Field>
        <Field label="Data e hora da observação" htmlFor="observed-at" error={form.formState.errors.observed_at?.message}>
          <TextInput id="observed-at" type="datetime-local" {...form.register("observed_at")} />
        </Field>
      </div>
      <div className="feature-grid">
        {featureKeys.map((feature) => (
          <div className="feature-field" key={feature}>
            <label htmlFor={`feature-${feature}`}>{feature}</label>
            <TextInput
              id={`feature-${feature}`}
              type="number"
              step="any"
              inputMode="decimal"
              placeholder="Ausente"
              aria-label={`${feature} — valor`}
              {...form.register(`values.${feature}`)}
            />
            {form.formState.errors.values?.[feature]?.message && (
              <span className="field-error">{form.formState.errors.values[feature]?.message}</span>
            )}
            <TextInput
              className="unit-input"
              maxLength={100}
              placeholder="Unidade opcional"
              aria-label={`${feature} — unidade opcional`}
              {...form.register(`units.${feature}`)}
            />
          </div>
        ))}
      </div>
      <Alert tone="warning" title="Dados e unidades">
        Campos em branco serão armazenados como ausentes; o backend poderá imputá-los na inferência.
        As unidades são apenas registradas, sem conversão ou validação. Confirme as unidades e faixas
        esperadas pelo protocolo do projeto antes de inserir dados.
      </Alert>
      <div className="form-actions">
        <Button type="submit" disabled={createObservation.isPending || episodes.length === 0}>
          {createObservation.isPending ? "Salvando..." : "Registrar observação"}
        </Button>
      </div>
    </form>
  );
}
