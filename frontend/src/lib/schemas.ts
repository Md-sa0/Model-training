import { z } from "zod";
import { featureKeys } from "../types/api";
import { toIsoWithTimezone } from "./format";

export const patientFormSchema = z.object({
  external_id: z.string().trim().max(200, "Máximo de 200 caracteres.").optional(),
  initial_episode_label: z.string().trim().max(200, "Máximo de 200 caracteres.").optional(),
  episode_started_at: z.string().optional(),
}).superRefine((form, context) => {
  if (form.episode_started_at && !toIsoWithTimezone(form.episode_started_at)) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["episode_started_at"],
      message: "Informe uma data e hora válidas.",
    });
  }
});

export const episodeFormSchema = z.object({
  label: z.string().trim().max(200, "Máximo de 200 caracteres.").optional(),
  started_at: z.string().optional(),
}).superRefine((form, context) => {
  if (form.started_at && !toIsoWithTimezone(form.started_at)) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["started_at"],
      message: "Informe uma data e hora válidas.",
    });
  }
});

const featureValuesSchema = z.record(z.string());

export const observationFormSchema = z.object({
  episode_id: z.string().uuid("Selecione um episódio válido."),
  observed_at: z.string().min(1, "Informe a data e hora da observação."),
  values: featureValuesSchema,
  units: featureValuesSchema,
}).superRefine((form, context) => {
  if (!toIsoWithTimezone(form.observed_at)) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["observed_at"],
      message: "Informe uma data e hora válidas.",
    });
  }
  for (const feature of featureKeys) {
    const value = form.values[feature]?.trim() ?? "";
    if (value !== "" && !Number.isFinite(Number(value))) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["values", feature],
        message: "Informe um número finito ou deixe em branco.",
      });
    }
    const unit = form.units[feature]?.trim() ?? "";
    if (unit.length > 100) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["units", feature],
        message: "Máximo de 100 caracteres.",
      });
    }
  }
});

export function toPatientPayload(input: z.infer<typeof patientFormSchema>) {
  const payload: {
    external_id?: string;
    initial_episode_label?: string;
    episode_started_at?: string;
  } = {};
  if (input.external_id?.trim()) payload.external_id = input.external_id.trim();
  if (input.initial_episode_label?.trim()) {
    payload.initial_episode_label = input.initial_episode_label.trim();
  }
  if (input.episode_started_at) {
    const iso = toIsoWithTimezone(input.episode_started_at);
    if (!iso) throw new Error("Data/hora inicial inválida.");
    payload.episode_started_at = iso;
  }
  return payload;
}

export function toEpisodePayload(input: z.infer<typeof episodeFormSchema>) {
  const payload: { label?: string; started_at?: string } = {};
  if (input.label?.trim()) payload.label = input.label.trim();
  if (input.started_at) {
    const iso = toIsoWithTimezone(input.started_at);
    if (!iso) throw new Error("Data/hora inválida.");
    payload.started_at = iso;
  }
  return payload;
}

export function buildObservationPayload(
  input: z.infer<typeof observationFormSchema>,
  observationId: string,
) {
  const values = Object.fromEntries(
    featureKeys.map((feature) => {
      const raw = input.values[feature]?.trim() ?? "";
      return [feature, raw === "" ? null : Number(raw)];
    }),
  ) as Record<(typeof featureKeys)[number], number | null>;
  const units = Object.fromEntries(
    featureKeys
      .map((feature) => [feature, input.units[feature]?.trim() ?? ""] as const)
      .filter(([, unit]) => unit !== ""),
  ) as Record<string, string>;
  const observedAt = toIsoWithTimezone(input.observed_at);
  if (!observedAt) throw new Error("Data/hora inválida.");
  return {
    observation_id: observationId,
    episode_id: input.episode_id,
    observed_at: observedAt,
    values,
    units,
  };
}
